import sys
from pathlib import Path

# Run this checkout even when another DiffSynth repository is installed editable.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch, os, json, ast
import numpy as np
from PIL import Image
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.dataset import RLinfDataset, SimpleVLARealWorldRLinfDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected true or false, got {value!r}")

# --- Patch Start: 允许加载包含 set 的权重文件 ---
try:
    torch.serialization.add_safe_globals(['set', 'OrderedDict', 'builtins.set'])
except AttributeError:
    pass
# --- Patch End ---



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        static_video_prob=0.0, # 新增参数
        action_dim=7,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(model_id=audio_processor_config.split(":")[0], origin_file_pattern=audio_processor_config.split(":")[1])
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, audio_processor_config=audio_processor_config)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.static_video_prob = static_video_prob # 保存参数
        
        
    def forward_preprocess(self, data):
        # === 新增：静态样本增强 ===
        # 如果启用，随机将当前样本变为“完全静止”，强迫模型学习背景保持
        if self.training and self.static_video_prob > 0 and np.random.rand() < self.static_video_prob:
            first_frame = data["video"][0]
            data["video"] = [first_frame] * len(data["video"])
            if "action" in data:
                data["action"] = torch.zeros_like(data["action"])
        # ============================
        # CFG-sensitive parameters
        # inputs_posi = {"prompt": data["prompt"]}
        inputs_posi = {}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "idx":data["idx"] if "idx" in data else None,
        }
        
        # Extra inputs
        # print(f'====WanTrainingModule forward_preprocess extra_inputs: {self.extra_inputs}====')
        # control_video, reference_image, etc.
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--static_video_prob", type=float, default=0.15, help="Probability of replacing the sample with a static video (action=0)")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval in epochs")
    parser.add_argument("--dataset",type=str,default="RLinfNpyDataset",help="Dataset type for training")
    parser.add_argument("--action_dim", type=int, default=7, help="Action dimension for dataset and hash-based WanModel config override.")
    parser.add_argument("--val_dataset_base_path", type=str, default="[]", help="Validation dataset base paths in JSON list format.")
    parser.add_argument("--train_dataset_base_path", type=str, default="[]", help="Training dataset base paths in JSON list format.")
    parser.add_argument("--Ta", type=int, default=8, help="Action prediction window length")
    parser.add_argument("--To", type=int, default=4, help="Observation context window length")
    parser.add_argument("--action2obs_bias", type=parse_bool, default=False, help="Whether to use action2obs bias")
    parser.add_argument("--retain_actions", type=parse_bool, default=True, help="Whether to retain context actions")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-window stride in frames")
    parser.add_argument("--max_finish_step", type=int, default=0, help="Trajectory end limit; 0 uses the full trajectory")
    parser.add_argument("--action_time_encoding", choices=["sinusoidal", "none"], default=os.environ.get("WAN_ACTION_TIME_ENCODING", "sinusoidal"), help="Action context time encoding; none reproduces legacy behavior")
    args = parser.parse_args()

    def _parse_path_list(arg_value, arg_name):
        if isinstance(arg_value, list):
            return arg_value
        if isinstance(arg_value, str):
            try:
                parsed = json.loads(arg_value)
            except json.JSONDecodeError:
                # Fallback for python-style list strings (e.g. trailing commas).
                try:
                    parsed = ast.literal_eval(arg_value)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(
                        f"{arg_name} must be a JSON/Python list string, got: {arg_value}"
                    ) from e
            if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
                raise ValueError(f"{arg_name} must be a JSON list of strings, got: {parsed}")
            return parsed
        raise ValueError(f"{arg_name} must be list[str] or JSON list string, got type {type(arg_value)}")

    args.train_dataset_base_path = _parse_path_list(args.train_dataset_base_path, "--train_dataset_base_path")
    args.val_dataset_base_path = _parse_path_list(args.val_dataset_base_path, "--val_dataset_base_path")
    os.environ["WAN_ACTION_DIM"] = str(args.action_dim)
    os.environ["WAN_ACTION_TIME_ENCODING"] = args.action_time_encoding
    if args.stride < 1 or args.Ta < 1 or args.To < 1 or args.max_finish_step < 0:
        parser.error("stride, Ta and To must be positive; max_finish_step must be nonnegative")
    dataset_kwargs = dict(Ta=args.Ta, To=args.To, stride=args.stride,
                          max_finish_step=args.max_finish_step,
                          retain_actions=args.retain_actions, action2obs_bias=args.action2obs_bias)

    if args.dataset == "RLinfDataset":
        dataset = RLinfDataset(
            base_path=args.train_dataset_base_path,
            repeat=args.dataset_repeat,
            action_dim=args.action_dim,
            **dataset_kwargs,
        )
        val_dataset = RLinfDataset(
            base_path=args.val_dataset_base_path,
            repeat=1,
            action_dim=args.action_dim,
            **dataset_kwargs,
        )
    elif args.dataset == "SimpleVLARealWorldRLinfDataset":
        dataset = SimpleVLARealWorldRLinfDataset(
            base_path=args.train_dataset_base_path,
            repeat=args.dataset_repeat,
            **dataset_kwargs,
        )
        val_dataset = SimpleVLARealWorldRLinfDataset(
            base_path=args.val_dataset_base_path,
            repeat=1,
            **dataset_kwargs,
        )
    else:
        raise NotImplementedError('this dataset type not implemented')
    # ----------------------
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        static_video_prob=args.static_video_prob, 
        action_dim=args.action_dim,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, val_dataset, model, model_logger, args=args)
