import json
import math
import random

import numpy as np
import wgpu
import wgpu.backends.wgpu_native

from wgpu_kernel import (
    BACKWARD_HY_SHADER,
    BPTT_STEP_SHADER,
    DX_EMBED_AND_UPDATE_SHADER,
    FINAL_LOGITS_SHADER,
    GRADIENT_RESET_SHADER,
    INITIAL_DH_SHADER,
    LOSS_SHADER,
    RNN_STEP_SHADER,
    SAMPLING_SHADER,
    SOFTMAX_DY_SHADER,
    SOFTMAX_SHADER,
    UPDATE_WEIGHTS_SHADER,
    create_compute_pipeline,
    device,
)

STORAGE_BUFFER_USAGE = wgpu.BufferUsage.STORAGE

# ==============================================================================
# OPTIMIZED MODEL CLASS (REFRACTORED AND CORRECTED)
# ==============================================================================


class SimpleRNN:
    # ----------------------------------------------------------------------
    # 1. CENTRALIZED BINDING CONFIGURATION (REFACTORED: Added resource type)
    # Maps (binding_index, buffer_resource_name, resource_type)
    # resource_type: 'ro' (read-only storage), 'rw' (read/write storage/atomic)
    # The last entry is always the uniform buffer if present.
    # The resource names must map to keys in self._resources or be passed
    # as overrides/specials (e.g., 'corpus_gpu', 'h_history_gpu').
    # ----------------------------------------------------------------------
    KERNEL_BINDINGS = {
        "rnn_step": {
            "shader": RNN_STEP_SHADER,
            "uniform": "rnn_step",
            "bindings": [
                (0, "W_embed", "ro"),
                (1, "W_xh", "ro"),
                (2, "W_hh", "ro"),
                (3, "b_h", "ro"),
                (4, "h_in", "ro"),
                (5, "h_out", "rw"),
                (6, "corpus_gpu", "ro"),
            ],
        },
        "final_logits": {
            "shader": FINAL_LOGITS_SHADER,
            "uniform": "final_logits",
            "bindings": [
                (0, "W_hy", "ro"),
                (1, "b_y", "ro"),
                (2, "h_in", "ro"),
                (3, "logits_out", "rw"),
            ],
        },
        "softmax_dy": {
            "shader": SOFTMAX_DY_SHADER,
            "uniform": "softmax_dy",
            "bindings": [
                (0, "logits_gpu", "ro"),
                (1, "dy_gpu", "rw"),
                (2, "temp_softmax_gpu", "rw"),
            ],
        },
        "backward_hy": {
            "shader": BACKWARD_HY_SHADER,
            "uniform": "backward_hy",
            "bindings": [
                (0, "dy_gpu", "ro"),
                (1, "h_final", "ro"),
                (2, "dW_hy", "rw"),
                (3, "db_y", "rw"),
            ],
        },
        "initial_dh": {
            "shader": INITIAL_DH_SHADER,
            "uniform": "backward_hy",  # Reuse uniform buffer
            "bindings": [
                (0, "W_hy", "ro"),
                (1, "dy_gpu", "ro"),
                (2, "dh_out", "rw"),
            ],
        },
        "bptt_step": {
            "shader": BPTT_STEP_SHADER,
            "uniform": "bptt_step",
            "bindings": [
                (0, "W_hh", "ro"),
                (1, "W_embed", "ro"),
                (2, "dh_next_in", "ro"),
                (3, "h_history_gpu", "ro"),
                (4, "dW_xh", "rw"),
                (5, "dW_hh", "rw"),
                (6, "db_h", "rw"),
                (7, "dh_next_out", "rw"),
                (8, "dh_raw_gpu", "rw"),
                (9, "corpus_gpu", "ro"),
            ],
        },
        "dx_embed_update": {
            "shader": DX_EMBED_AND_UPDATE_SHADER,
            "uniform": "dx_embed_update",
            "bindings": [
                (0, "W_xh", "ro"),
                (1, "dh_raw_gpu", "ro"),
                (2, "dW_embed", "rw"),
                (3, "corpus_gpu", "ro"),
            ],
        },
        "update_weights": {
            "shader": UPDATE_WEIGHTS_SHADER,
            "uniform": "update_weights",
            "bindings": [
                (0, "param_buf", "rw"),
                (1, "grad_buf", "ro"),
            ],
        },
        "loss": {
            "shader": LOSS_SHADER,
            "uniform": "loss",
            "bindings": [
                (0, "logits_gpu", "ro"),
                (1, "loss_out_gpu", "rw"),
                (
                    2,
                    "temp_storage_gpu_loss",
                    "rw",
                ),
            ],
        },
        "softmax": {
            "shader": SOFTMAX_SHADER,
            "uniform": "softmax",
            "bindings": [
                (0, "logits_gpu", "ro"),
                (
                    1,
                    "probs_gpu",
                    "rw",
                ),  # FIX: Changed from 'probs_out' to 'probs_gpu' for consistency with self._resources
                (2, "temp_softmax_gpu", "rw"),
            ],
        },
        "sampling": {
            "shader": SAMPLING_SHADER,
            "uniform": "sampling",
            "bindings": [
                (0, "probs_gpu", "ro"),
                (1, "out_idx_gpu", "rw"),
            ],
        },
        "gradient_reset": {
            "shader": GRADIENT_RESET_SHADER,
            "uniform": None,  # No uniform needed
            "bindings": [
                # This binding will be replaced for each gradient buffer:
                # dW_xh_atomic_gpu, dW_hh_atomic_gpu, dW_hy_atomic_gpu, etc.
                (0, "grad_buffer", "rw"),
            ],
        },
    }

    def __init__(self, vocab_size, hidden_size, embedding_dim):
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        H, V, E = hidden_size, vocab_size, embedding_dim

        # Alignment and padding logic (Unchanged)
        self.alignment = device.limits["min-storage-buffer-offset-alignment"]
        unpadded_h_size = self.hidden_size * 4  # size in bytes
        self.padded_h_size = (
            (unpadded_h_size + self.alignment - 1) // self.alignment * self.alignment
        )
        print(f"INFO: Using storage buffer alignment of {self.alignment} bytes.")
        print(
            f"INFO: Hidden state size padded from {unpadded_h_size} to {self.padded_h_size} bytes."
        )

        # Parameter Initialization (Unchanged structure)
        self.params = {
            "W_embed": np.random.uniform(-0.01, 0.01, (E, V)).astype(np.float32),
            "W_xh": np.random.uniform(-0.01, 0.01, (H, E)).astype(np.float32),
            "W_hh": np.random.uniform(-0.01, 0.01, (H, H)).astype(np.float32),
            "W_hy": np.random.uniform(-0.01, 0.01, (V, H)).astype(np.float32),
            "b_h": np.zeros(H, dtype=np.float32),
            "b_y": np.zeros(V, dtype=np.float32),
        }

        self.params_gpu = {}
        self.grads_gpu = {}
        param_usage = (
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_DST
            | wgpu.BufferUsage.COPY_SRC
        )
        grad_usage = (
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_DST
            | wgpu.BufferUsage.COPY_SRC
        )

        # --- CORRECTED GRADIENT BUFFER CREATION ---
        GRAD_MAP = {
            "W_embed": "dW_embed",
            "W_xh": "dW_xh",
            "W_hh": "dW_hh",
            "W_hy": "dW_hy",
            "b_h": "db_h",
            "b_y": "db_y",
        }

        for name, arr in self.params.items():
            self.params_gpu[name] = device.create_buffer_with_data(
                data=arr, usage=param_usage
            )
            grad_size = (arr.nbytes + 3) // 4 * 4

            # Use the explicit map to ensure correct keys (e.g., dW_hy, db_y) are created
            grad_name = GRAD_MAP[name]
            self.grads_gpu[grad_name] = device.create_buffer(
                size=grad_size, usage=grad_usage
            )

        # NEW: List of gradient buffers for the GPU reset function
        self.gradient_buffers = list(self.grads_gpu.values())
        # --- END GRADIENT BUFFER CREATION ---

        # Temporary/Shared Buffers (UNCHANGED)
        shared_usage = (
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST
        )
        dh_usage = shared_usage

        self.h_fwd_ping = device.create_buffer(size=H * 4, usage=shared_usage)
        self.h_fwd_pong = device.create_buffer(size=H * 4, usage=shared_usage)
        self.zero_buffer(self.h_fwd_ping)
        self.zero_buffer(self.h_fwd_pong)

        self.gen_h_out = device.create_buffer(
            size=H * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        self.gen_input = device.create_buffer(
            size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self.logits_gpu = device.create_buffer(size=V * 4, usage=shared_usage)

        self.dy_gpu = device.create_buffer(size=V * 4, usage=shared_usage)
        self.temp_softmax_gpu = device.create_buffer(size=2 * 4, usage=shared_usage)

        # NEW: Temporary buffer for the parallel LOSS calculation (Kernel 9)
        self.temp_storage_gpu_loss = device.create_buffer(
            size=2 * 4, usage=shared_usage
        )

        self.dh_ping = device.create_buffer(size=H * 4, usage=dh_usage)
        self.dh_pong = device.create_buffer(size=H * 4, usage=dh_usage)

        self.dh_raw_gpu = device.create_buffer(size=H * 4, usage=shared_usage)
        self.probs_gpu = device.create_buffer(size=V * 4, usage=shared_usage)
        self.out_idx_gpu = device.create_buffer(size=4, usage=shared_usage)

        # --- NEW: Resource Mapping for all buffers (Coupling reduction) ---
        self._resources = {}
        self._resources.update(self.params_gpu)
        self._resources.update(self.grads_gpu)
        self._resources.update(
            {
                "h_fwd_ping": self.h_fwd_ping,
                "h_fwd_pong": self.h_fwd_pong,
                "gen_h_out": self.gen_h_out,
                "gen_input": self.gen_input,
                "logits_gpu": self.logits_gpu,
                "dy_gpu": self.dy_gpu,
                "temp_softmax_gpu": self.temp_softmax_gpu,
                "temp_storage_gpu_loss": self.temp_storage_gpu_loss,  # NEW BINDING
                "dh_ping": self.dh_ping,
                "dh_pong": self.dh_pong,
                "dh_raw_gpu": self.dh_raw_gpu,
                "probs_gpu": self.probs_gpu,
                "out_idx_gpu": self.out_idx_gpu,
            }
        )
        # --- END NEW: Resource Mapping ---

        # Uniform Buffers (Unchanged structure)
        self.uniform_buffers = {}
        uniform_specs = {
            "rnn_step": 4 * 4,
            "final_logits": 2 * 4,
            "softmax_dy": 2 * 4,
            "backward_hy": 2 * 4,
            "bptt_step": 6 * 4,
            "dx_embed_update": 4 * 4,
            "update_weights": 4,
            "loss": 2 * 4,
            "softmax": 4,
            "sampling": 8,
        }
        uniform_usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        for name, size in uniform_specs.items():
            self.uniform_buffers[name] = device.create_buffer(
                size=size, usage=uniform_usage
            )

        self._create_pipelines()

    def _create_pipelines(self):
        s = wgpu.ShaderStage
        ro, rw, u = (
            {"type": "read-only-storage"},
            {"type": "storage"},
            {"type": "uniform"},
        )
        # Map the shorthand 'ro'/'rw'/'u' from KERNEL_BINDINGS to the wgpu resource dictionary
        buffer_type_map = {"ro": ro, "rw": rw, "u": u}

        self.bgls = {}
        for name, config in self.KERNEL_BINDINGS.items():
            entries = []

            # Use the unified binding config to automatically build BGL entries
            for binding_idx, _, resource_type_key in config["bindings"]:
                buffer_type = buffer_type_map.get(resource_type_key)
                if not buffer_type:
                    raise ValueError(f"Unknown resource type key: {resource_type_key}")
                entries.append(
                    {
                        "binding": binding_idx,
                        "visibility": s.COMPUTE,
                        "buffer": buffer_type,
                    }
                )

            # Append the uniform binding ONLY IF a uniform name is provided
            uniform_name = config.get("uniform")  # FIX: Get the uniform name
            if uniform_name is not None:  # FIX: Check if the uniform name is not None
                entries.append(
                    {
                        "binding": len(
                            config["bindings"]
                        ),  # Uniform is always the binding immediately after the storage buffers
                        "visibility": s.COMPUTE,
                        "buffer": u,
                    }
                )

            self.bgls[name] = self.device.create_bind_group_layout(entries=entries)

        # Pipeline creation remains the same (just mapping kernel names to pipelines)
        self.pipelines = {
            "rnn_step": create_compute_pipeline(
                RNN_STEP_SHADER, [self.bgls["rnn_step"]]
            ),
            "final_logits": create_compute_pipeline(
                FINAL_LOGITS_SHADER, [self.bgls["final_logits"]]
            ),
            "softmax_dy_reductions": create_compute_pipeline(
                SOFTMAX_DY_SHADER, [self.bgls["softmax_dy"]], "compute_reductions"
            ),
            "softmax_dy_compute": create_compute_pipeline(
                SOFTMAX_DY_SHADER, [self.bgls["softmax_dy"]], "compute_dy"
            ),
            "backward_hy": create_compute_pipeline(
                BACKWARD_HY_SHADER, [self.bgls["backward_hy"]]
            ),
            "initial_dh": create_compute_pipeline(
                INITIAL_DH_SHADER, [self.bgls["initial_dh"]]
            ),
            "bptt_step": create_compute_pipeline(
                BPTT_STEP_SHADER, [self.bgls["bptt_step"]]
            ),
            "dx_embed_update": create_compute_pipeline(
                DX_EMBED_AND_UPDATE_SHADER, [self.bgls["dx_embed_update"]]
            ),
            "update_weights": create_compute_pipeline(
                UPDATE_WEIGHTS_SHADER, [self.bgls["update_weights"]]
            ),
            "loss": create_compute_pipeline(LOSS_SHADER, [self.bgls["loss"]]),
            "softmax_reductions": create_compute_pipeline(
                SOFTMAX_SHADER, [self.bgls["softmax"]], "compute_reductions"
            ),
            "softmax_compute": create_compute_pipeline(
                SOFTMAX_SHADER, [self.bgls["softmax"]], "compute_probs"
            ),
            "sampling": create_compute_pipeline(
                SAMPLING_SHADER, [self.bgls["sampling"]]
            ),
            "gradient_reset": create_compute_pipeline(
                GRADIENT_RESET_SHADER, [self.bgls["gradient_reset"]]
            ),
        }

    # ----------------------------------------------------------------------
    # 2. BIND GROUP HELPER
    # Uses KERNEL_BINDINGS and _resources to create the BindGroup
    # ----------------------------------------------------------------------
    def _create_bind_group(self, pipeline_name, overrides=None):
        """Creates a BindGroup for a given pipeline using KERNEL_BINDINGS and overrides."""
        config = self.KERNEL_BINDINGS[pipeline_name]
        bgl = self.bgls[pipeline_name]
        overrides = overrides or {}
        entries = []

        # Resolve all storage buffer bindings
        # Unpack the 3-element tuple (idx, name, type), ignoring type for the BindGroup
        for binding_idx, resource_name, _ in config["bindings"]:
            # Priority: 1. Overrides (runtime buffers) 2. self._resources (params/grads/temps)
            buffer_resource = overrides.get(
                resource_name, self._resources.get(resource_name)
            )

            if isinstance(buffer_resource, tuple):
                # Handle (gpu_buffer, offset, size) tuple for manual offsets
                buf, offset, size = buffer_resource
                entries.append(
                    {
                        "binding": binding_idx,
                        "resource": {"buffer": buf, "offset": offset, "size": size},
                    }
                )
            elif buffer_resource is not None:
                entries.append(
                    {
                        "binding": binding_idx,
                        "resource": {"buffer": buffer_resource},
                    }
                )
            else:
                raise ValueError(
                    f"Resource '{resource_name}' for pipeline '{pipeline_name}' not found in resources or overrides."
                )

        # Resolve Uniform buffer binding
        if "uniform" in config:
            uniform_name = config["uniform"]
            if uniform_name is not None:
                entries.append(
                    {
                        "binding": len(config["bindings"]),
                        "resource": {"buffer": self.uniform_buffers[uniform_name]},
                    }
                )

        return self.device.create_bind_group(layout=bgl, entries=entries)

    # ----------------------------------------------------------------------
    # 3. DISPATCH HELPER (REDUCED COUPLING)
    # ----------------------------------------------------------------------
    def _dispatch_compute_job(
        self,
        command_encoder,
        pipeline_name,
        uniform_data=None,
        workgroups_x=1,
        workgroups_y=1,
        overrides=None,
        pre_created_bind_group=None,
        dynamic_offsets=None,
    ):
        """
        Helper function to handle the boilerplate. Uses the new _create_bind_group
        for single-shot dispatches.
        """

        config = self.KERNEL_BINDINGS[pipeline_name]

        if "uniform" in config and uniform_data is not None:
            # Write uniform data first
            uniform_name = config["uniform"]
            self.device.queue.write_buffer(
                self.uniform_buffers[uniform_name], 0, uniform_data.tobytes()
            )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines[pipeline_name])

        if pre_created_bind_group:
            # Case 1: BindGroup is pre-created outside a loop
            compute_pass.set_bind_group(
                0, pre_created_bind_group, dynamic_offsets_data=dynamic_offsets or []
            )
        else:
            # Case 2: BindGroup must be created for a single, non-loop dispatch
            bind_group = self._create_bind_group(pipeline_name, overrides)
            compute_pass.set_bind_group(0, bind_group)

        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y)
        compute_pass.end()

    # --- Utility methods (Mostly Unchanged) ---
    def get_initial_hidden_state_gpu(self):
        usage = (
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST
        )
        h_buffer = device.create_buffer(size=self.hidden_size * 4, usage=usage)
        self.zero_buffer(h_buffer)
        return h_buffer

    def zero_buffer(self, buffer, offset=0, size=None):
        if size is None:
            size = buffer.size - offset
        command_encoder = device.create_command_encoder()
        command_encoder.clear_buffer(buffer, offset, size)
        device.queue.submit([command_encoder.finish()])

    def update_hidden_state(self, h_source, h_dest):
        S = (h_source.size // self.padded_h_size) - 1
        command_encoder = device.create_command_encoder()
        command_encoder.copy_buffer_to_buffer(
            h_source, S * self.padded_h_size, h_dest, 0, h_dest.size
        )
        device.queue.submit([command_encoder.finish()])

    def _get_params(self):
        cpu_params = {}
        for name, gpu_buf in self.params_gpu.items():
            data = device.queue.read_buffer(gpu_buf).cast("f")
            arr = np.array(data).reshape(self.params[name].shape)
            cpu_params[name] = arr.tolist()
        return cpu_params

    def _set_params(self, params):
        for name, data in params.items():
            arr = np.array(data, dtype=np.float32)
            self.params[name] = arr
            device.queue.write_buffer(self.params_gpu[name], 0, arr.tobytes())

    def save_model(self, filepath: str):
        model_data = {
            "config": {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "embedding_dim": self.embedding_dim,
            },
            "params": self._get_params(),
        }
        with open(filepath, "w") as f:
            json.dump(model_data, f)
        print(f"INFO: Model saved to '{filepath}'")

    @classmethod
    def load_model(cls, filepath: str):
        print(f"INFO: Loading model from '{filepath}'...")
        with open(filepath, "r") as f:
            model_data = json.load(f)
        config = model_data["config"]
        model = cls(
            config["vocab_size"], config["hidden_size"], config["embedding_dim"]
        )
        model._set_params(model_data["params"])
        return model

    # ----------------------------------------------------------------------
    # 4. FORWARD SEQUENCE (CLEANED LOOPS - UNCHANGED FROM PREVIOUS REFACTOR)
    # ----------------------------------------------------------------------
    def forward_sequence(self, corpus_gpu, h_prev_gpu, offset, seq_length):
        S, H, V = seq_length, self.hidden_size, self.vocab_size
        H_bytes = H * 4

        h_history_gpu = self.device.create_buffer(
            size=(S + 1) * self.padded_h_size,
            usage=wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST,
        )

        h_ping = self.h_fwd_ping
        h_pong = self.h_fwd_pong

        command_encoder = self.device.create_command_encoder()
        command_encoder.copy_buffer_to_buffer(h_prev_gpu, 0, h_ping, 0, H_bytes)
        command_encoder.copy_buffer_to_buffer(h_ping, 0, h_history_gpu, 0, H_bytes)

        # --- OPTIMIZATION: Pre-create BindGroups (Ping->Pong and Pong->Ping) ---
        # Use the abstract helper, passing runtime/loop-specific buffers as overrides.
        bg_rnn_a = self._create_bind_group(
            "rnn_step",
            overrides={"h_in": h_ping, "h_out": h_pong, "corpus_gpu": corpus_gpu},
        )
        bg_rnn_b = self._create_bind_group(
            "rnn_step",
            overrides={"h_in": h_pong, "h_out": h_ping, "corpus_gpu": corpus_gpu},
        )

        current_bg = bg_rnn_a
        # --- END OPTIMIZATION ---

        for t in range(seq_length):
            uniform_data = np.array(
                [H, self.embedding_dim, V, offset + t], dtype=np.uint32
            )

            self._dispatch_compute_job(
                command_encoder,
                "rnn_step",
                uniform_data=uniform_data,
                workgroups_x=math.ceil(H / 64),
                pre_created_bind_group=current_bg,
            )

            h_out_buf = h_pong if current_bg == bg_rnn_a else h_ping

            command_encoder.copy_buffer_to_buffer(
                h_out_buf,
                0,
                h_history_gpu,
                (t + 1) * self.padded_h_size,
                H_bytes,
            )

            current_bg = bg_rnn_b if current_bg == bg_rnn_a else bg_rnn_a

        final_h_gpu = h_ping if current_bg == bg_rnn_b else h_pong

        # Final logits calculation
        self._dispatch_compute_job(
            command_encoder,
            "final_logits",
            uniform_data=np.array([H, V], dtype=np.uint32),
            workgroups_x=math.ceil(V / 64),
            overrides={"h_in": final_h_gpu, "logits_out": self.logits_gpu},
        )

        self.device.queue.submit([command_encoder.finish()])

        return self.logits_gpu, h_history_gpu

    # ----------------------------------------------------------------------
    # 5. BACKWARD SEQUENCE (FIXED: Removed redundant explicit grad lookups)
    # ----------------------------------------------------------------------
    def calculate_loss_gpu(self, logits_gpu, target_idx):
        V = self.vocab_size

        # New: Temporary storage for parallel reduction (2 floats: max_val, sum_val)
        temp_storage_gpu_loss = self.device.create_buffer(
            size=8, usage=wgpu.BufferUsage.STORAGE
        )

        loss_out_gpu = self.device.create_buffer(
            size=4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        command_encoder = self.device.create_command_encoder()

        # Workgroup size is 256. Need V / 256 + 1 workgroup for full coverage.
        # Since the new LOSS_SHADER performs a parallel reduction where one workgroup
        # covers the entire V, we only dispatch a single workgroup (WG_SIZE=256)
        # to ensure local memory is shared correctly.
        # NOTE: This assumes V is large enough to warrant parallel reduction.
        # If V > 256, it still only needs one workgroup.
        workgroups_x = 1

        self._dispatch_compute_job(
            command_encoder,
            "loss",
            uniform_data=np.array([V, target_idx], dtype=np.uint32),
            workgroups_x=workgroups_x,
            overrides={
                "logits_gpu": logits_gpu,
                "loss_out_gpu": loss_out_gpu,
                "temp_storage_gpu_loss": temp_storage_gpu_loss,  # NEW binding
            },
        )
        self.device.queue.submit([command_encoder.finish()])
        return self.device.queue.read_buffer(loss_out_gpu).cast("f")[0]

    def reset_gradients_gpu(self, grad_buffers_to_reset):
        """
        Efficiently resets all atomic gradient buffers to 0.0 using the GPU.
        :param grad_buffers_to_reset: A list of WGPU Buffer objects to reset.
        """
        command_encoder = self.device.create_command_encoder()

        for grad_buffer in grad_buffers_to_reset:
            buffer_size_floats = grad_buffer.size // 4
            # Workgroup size is 256.
            workgroups_x = (buffer_size_floats + 255) // 256

            self._dispatch_compute_job(
                command_encoder,
                "gradient_reset",
                uniform_data=None,  # No uniforms for this kernel
                workgroups_x=workgroups_x,
                overrides={"grad_buffer": grad_buffer},
            )

        self.device.queue.submit([command_encoder.finish()])
        # NOTE: If this is called once per training step, it should be fast.

    def backward_sequence(
        self, corpus_gpu, offset, seq_length, target_idx, logits_gpu, h_history_gpu
    ):
        S, H, V = seq_length, self.hidden_size, self.vocab_size
        H_bytes = H * 4

        command_encoder = self.device.create_command_encoder()
        for grad_buf in self.grads_gpu.values():
            command_encoder.clear_buffer(grad_buf)

        # Softmax dy calculation
        uniform_data_softmax = np.array([V, target_idx], dtype=np.uint32)
        bg_softmax = self._create_bind_group(
            "softmax_dy", overrides={"logits_gpu": logits_gpu}
        )

        self.device.queue.write_buffer(
            self.uniform_buffers["softmax_dy"], 0, uniform_data_softmax.tobytes()
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines["softmax_dy_reductions"])
        compute_pass.set_bind_group(0, bg_softmax)
        compute_pass.dispatch_workgroups(1)
        compute_pass.set_pipeline(self.pipelines["softmax_dy_compute"])
        compute_pass.set_bind_group(0, bg_softmax)
        compute_pass.dispatch_workgroups(math.ceil(V / 256))
        compute_pass.end()

        # Backward HY (FIXED: Removed dW_hy, db_y overrides)
        uniform_data_hy = np.array([H, V], dtype=np.uint32)
        final_h_offset = S * self.padded_h_size
        h_final_res = (h_history_gpu, final_h_offset, H_bytes)

        self._dispatch_compute_job(
            command_encoder,
            "backward_hy",
            uniform_data=uniform_data_hy,
            workgroups_x=math.ceil(V / 8),
            workgroups_y=math.ceil(H / 8),
            overrides={
                "h_final": h_final_res
            },  # Only h_final needs overriding (due to offset)
        )

        # Initial DH
        dh_ping = self.dh_ping
        dh_pong = self.dh_pong
        command_encoder.clear_buffer(dh_ping, 0, H_bytes)

        self._dispatch_compute_job(
            command_encoder,
            "initial_dh",
            workgroups_x=math.ceil(H / 64),
            pre_created_bind_group=self._create_bind_group(
                "initial_dh", overrides={"dh_out": dh_ping}
            ),
            uniform_data=uniform_data_hy,  # Re-uses backward_hy uniform
        )

        # BPTT Loop Setup
        bg_bptt_a = self._create_bind_group(
            "bptt_step",
            overrides={
                "dh_next_in": dh_ping,
                "dh_next_out": dh_pong,
                "h_history_gpu": h_history_gpu,
                "corpus_gpu": corpus_gpu,
            },
        )
        bg_bptt_b = self._create_bind_group(
            "bptt_step",
            overrides={
                "dh_next_in": dh_pong,
                "dh_next_out": dh_ping,
                "h_history_gpu": h_history_gpu,
                "corpus_gpu": corpus_gpu,
            },
        )
        current_bg = bg_bptt_a

        for t in reversed(range(seq_length)):
            uniform_data_bptt = np.array(
                [H, self.embedding_dim, V, t, offset, self.padded_h_size],
                dtype=np.uint32,
            )

            # BPTT Step
            self._dispatch_compute_job(
                command_encoder,
                "bptt_step",
                uniform_data=uniform_data_bptt,
                workgroups_x=math.ceil(H / 64),
                pre_created_bind_group=current_bg,
            )

            # DX_EMBED_UPDATE Step
            self._dispatch_compute_job(
                command_encoder,
                "dx_embed_update",
                uniform_data=np.array(
                    [H, self.embedding_dim, V, offset + t], dtype=np.uint32
                ),
                workgroups_x=math.ceil(self.embedding_dim / 64),
                overrides={"corpus_gpu": corpus_gpu},
            )

            current_bg = bg_bptt_b if current_bg == bg_bptt_a else bg_bptt_a

        self.device.queue.submit([command_encoder.finish()])

    # ----------------------------------------------------------------------
    # 6. UPDATE WEIGHTS (CLEANED LOOP)
    # ----------------------------------------------------------------------

    def update_weights(self, learning_rate):
        command_encoder = self.device.create_command_encoder()

        uniform_data = np.array([learning_rate], dtype=np.float32)
        update_weights_uniform_buffer = self.uniform_buffers["update_weights"]
        self.device.queue.write_buffer(
            update_weights_uniform_buffer, 0, uniform_data.tobytes()
        )

        # The GRAD_MAP must be replicated or the gradient names used must be derived
        # based on the static names. Since the params are fixed, we use a simple list
        # based on the fixed keys (W_embed, dW_embed, W_xh, dW_xh, etc.).
        GRAD_MAP = {
            "W_embed": "dW_embed",
            "W_xh": "dW_xh",
            "W_hh": "dW_hh",
            "W_hy": "dW_hy",
            "b_h": "db_h",
            "b_y": "db_y",
        }

        for name in self.params:
            param_buf = self.params_gpu[name]

            # Use the explicit map from __init__
            grad_name = GRAD_MAP[name]
            grad_buf = self.grads_gpu[grad_name]

            self._dispatch_compute_job(
                command_encoder,
                "update_weights",
                uniform_data=None,
                workgroups_x=math.ceil(param_buf.size / 4 / 256),
                overrides={"param_buf": param_buf, "grad_buf": grad_buf},
            )

        self.device.queue.submit([command_encoder.finish()])

    # ----------------------------------------------------------------------
    # 7. GENERATE STEP (CLEANED LOOPS - UNCHANGED FROM PREVIOUS REFACTOR)
    # ----------------------------------------------------------------------
    def forward_step(self, input_idx, h_gpu):
        H, E, V = self.hidden_size, self.embedding_dim, self.vocab_size
        self.device.queue.write_buffer(
            self.gen_input, 0, np.array([input_idx], dtype=np.uint32)
        )
        command_encoder = self.device.create_command_encoder()

        # 1. Forward Pass RNN Step
        self._dispatch_compute_job(
            command_encoder,
            "rnn_step",
            uniform_data=np.array([H, E, V, 0], dtype=np.uint32),
            workgroups_x=math.ceil(H / 64),
            overrides={
                "h_in": h_gpu,
                "h_out": self.gen_h_out,
                "corpus_gpu": self.gen_input,
            },
        )

        command_encoder.copy_buffer_to_buffer(self.gen_h_out, 0, h_gpu, 0, h_gpu.size)
        self.device.queue.submit([command_encoder.finish()])
        return h_gpu

    def generate_step(self, input_idx, h_gpu):
        H, V = self.hidden_size, self.vocab_size
        self.device.queue.write_buffer(
            self.gen_input, 0, np.array([input_idx], dtype=np.uint32)
        )
        command_encoder = self.device.create_command_encoder()

        # 1. Forward Pass RNN Step
        self._dispatch_compute_job(
            command_encoder,
            "rnn_step",
            uniform_data=np.array([H, self.embedding_dim, V, 0], dtype=np.uint32),
            workgroups_x=math.ceil(H / 64),
            overrides={
                "h_in": h_gpu,
                "h_out": self.gen_h_out,
                "corpus_gpu": self.gen_input,
            },
        )
        command_encoder.copy_buffer_to_buffer(self.gen_h_out, 0, h_gpu, 0, h_gpu.size)

        # 2. Final Logits
        self._dispatch_compute_job(
            command_encoder,
            "final_logits",
            uniform_data=np.array([H, V], dtype=np.uint32),
            workgroups_x=math.ceil(V / 64),
            overrides={"h_in": h_gpu, "logits_out": self.logits_gpu},
        )

        # 3. Softmax
        uniform_softmax_data = np.array([V], dtype=np.uint32)
        bg_softmax = self._create_bind_group("softmax")

        self.device.queue.write_buffer(
            self.uniform_buffers["softmax"], 0, uniform_softmax_data.tobytes()
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipelines["softmax_reductions"])
        compute_pass.set_bind_group(0, bg_softmax)
        compute_pass.dispatch_workgroups(1)
        compute_pass.set_pipeline(self.pipelines["softmax_compute"])
        compute_pass.set_bind_group(0, bg_softmax)
        compute_pass.dispatch_workgroups(math.ceil(V / 256))
        compute_pass.end()

        # 4. Sampling
        rand_u = random.random()
        uniform_data_sample = np.zeros(2, dtype=np.uint32)
        uniform_data_sample[0] = V
        uniform_data_sample.view(np.float32)[1] = rand_u

        self._dispatch_compute_job(
            command_encoder,
            "sampling",
            uniform_data=uniform_data_sample,
            workgroups_x=1,
            overrides={"probs_gpu": self.probs_gpu, "out_idx_gpu": self.out_idx_gpu},
        )

        self.device.queue.submit([command_encoder.finish()])

        return self.device.queue.read_buffer(self.out_idx_gpu).cast("I")[0]
