from ..models.wan_video_dit import WanModel
from ..models.wan_video_dit_s2v import WanS2VModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vae import WanVideoVAE, WanVideoVAE38
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..models.wan_video_mot import MotWanModel

model_loader_configs = [
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    # 新加的
    
    (None, "3803bd491e969952cbbeb1a48fd8fa11", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "d9c91561d7feb8510857001e88c410db", ["wan_video_dit"], [WanModel], "civitai"), # 14B I2V 
    (None, "16266d0dbcda81a4f797d7b8aafe0f35", ["wan_video_dit"], [WanModel], "civitai"), # 14B I2V control context lora
    (None, "dd9ee3cc97642977beab8e12d2610a19", ["wan_video_dit"], [WanModel], "civitai"), # 14B I2V control context lora 最新
    (None, "0c81d47223c50d8d74106f79310ae207", ["wan_video_dit"], [WanModel], "civitai"), # 5B TI2V action
    (None, "3f4e37438f72ef88cd27b161fd1b193c", ["wan_video_dit"], [WanModel], "civitai"), # 5B TI2V action
    (None, "d3abb829857dff2d9129d2f396a7eace", ["wan_video_dit"], [WanModel], "civitai"), # 5B TI2V action
    (None, "bc4824aef7c3f23d3378cec6e2b1316c", ["wan_video_dit"], [WanModel], "civitai"), # 5B TI2V action
    (None, "cd86b8137f89754c6c4557e285274a95", ["wan_video_dit"], [WanModel], "civitai"), # 5B TI2V ORCA 58D full checkpoint
    (None, "fcc43a93949201bafeb34aa1eb8bc50f", ["wan_video_dit"], [WanModel], "civitai"), # 5B TI2V action (agx/action_dim=10)
    (None, "6efc7e19e87c1755f17dec14be3d0bf1", ["wan_video_dit"], [WanModel], "civitai"), # 5B TI2V action
    #
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"), # 14B I2V
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "70ddad9d3a133785da5ea371aae09504", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "26bde73488a92e64cc20b0a7485b9e5b", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "ac6a5aa74f4a0aab6f64eb9a72f19901", ["wan_video_dit"], [WanModel], "civitai"), 
    (None, "b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [WanModel], "civitai"), 
    (None, "1f5ab7703c6fc803fdded85ff040c316", ["wan_video_dit"], [WanModel], "civitai"), #5B TI2V
    (None, "5b013604280dd715f8457c6ed6d6a626", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "2267d489f0ceb9f21836532952852ee5", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5ec04e02b42d2580483ad69f4e76346a", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "47dbeab5e560db3180adf51dc0232fb1", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5f90e66a0672219f12d9a626c8c21f61", ["wan_video_dit", "wan_video_vap"], [WanModel,MotWanModel], "diffusers"),
    (None, "a61453409b67cd3246cf0c3bebad47ba", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "7a513e1f257a861512b1afd387a8ecd9", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    (None, "966cffdcc52f9c46c391768b27637614", ["wan_video_dit"], [WanS2VModel], "civitai"),
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "e1de6c02cdac79f8b739f4d3698cd216", ["wan_video_vae"], [WanVideoVAE38], "civitai"),
    (None, "dbd5ec76bbf977983f972c151d545389", ["wan_video_motion_controller"], [WanMotionControllerModel], "civitai"),
    # (None, "d30fb9e02b1dbf4e509142f05cf7dd50", ["flux_dit", "step1x_connector"], [FluxDiT, Qwen2Connector], "civitai"),
    # (None, "30143afb2dea73d1ac580e0787628f8c", ["flux_lora_patcher"], [FluxLoraPatcher], "civitai"),
    # (None, "77c2e4dd2440269eb33bfaa0d004f6ab", ["flux_lora_encoder"], [FluxLoRAEncoder], "civitai"),
    # (None, "3e6c61b0f9471135fc9c6d6a98e98b6d", ["flux_dit", "nexus_gen_generation_adapter"], [FluxDiT, NexusGenAdapter], "civitai"),
    # (None, "63c969fd37cce769a90aa781fbff5f81", ["flux_dit", "nexus_gen_editing_adapter"], [FluxDiT, NexusGenImageEmbeddingMerger], "civitai"),
    # (None, "2bd19e845116e4f875a0a048e27fc219", ["nexus_gen_llm"], [NexusGenAutoregressiveModel], "civitai"),
    # (None, "0319a1cb19835fb510907dd3367c95ff", ["qwen_image_dit"], [QwenImageDiT], "civitai"),
    # (None, "8004730443f55db63092006dd9f7110e", ["qwen_image_text_encoder"], [QwenImageTextEncoder], "diffusers"),
    # (None, "ed4ea5824d55ec3107b09815e318123a", ["qwen_image_vae"], [QwenImageVAE], "diffusers"),
    # (None, "073bce9cf969e317e5662cd570c3e79c", ["qwen_image_blockwise_controlnet"], [QwenImageBlockWiseControlNet], "civitai"),
    # (None, "a9e54e480a628f0b956a688a81c33bab", ["qwen_image_blockwise_controlnet"], [QwenImageBlockWiseControlNet], "civitai"),
    (None, "31fa352acb8a1b1d33cd8764273d80a2", ["wan_video_dit", "wan_video_animate_adapter"], [WanModel, WanAnimateAdapter], "civitai"),
]
