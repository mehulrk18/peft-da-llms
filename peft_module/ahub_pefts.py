from enum import Enum

# from adapters import LoRAConfig, SeqBnConfig, DoubleSeqBnConfig, ParBnConfig, CompacterConfig, \
#     CompacterPlusPlusConfig #IA3Config
from peft import LoraConfig, IA3Config, LoKrConfig, AdaLoraConfig, AdaptionPromptConfig, LoHaConfig, OFTConfig


class PEFTEnum(Enum):
    BITFIT = "bitfit"
    LORA = "lora"
    IA3 = "ia3"
    LO_REFT = "loreft"
    NO_REFT = "noreft"
    DI_REFT = "direft"
    SEQUENCE_BN = "seq_bn"
    DOUBLE_SEQUENCE_BN = "double_seq_bn"
    PARALLEL_BN = "par_bn"
    COMPACTER = "compacter"
    COMPACTER_PP = "compacter++"
    LoKr = "lokr"
    LoHa = "loha"
    AdaLoRA = "adalora"
    LlamaAdapter = "llama_adapter"
    OFT = "oft"
    ReFT = "reft"


pefts_configuration = {
    "ah": {
        # PEFTEnum.BITFIT.name: NotImplemented, # see how to implement
        # PEFTEnum.LORA.name: LoRAConfig,
        # # PEFTEnum.IA3.name: IA3Config,
        # PEFTEnum.LO_REFT.name: NotImplemented, # ReftConfig, # use pyreft
        # PEFTEnum.NO_REFT.name: NotImplemented,
        # PEFTEnum.DI_REFT.name: NotImplemented,
        # PEFTEnum.SEQUENCE_BN.name: SeqBnConfig,
        # PEFTEnum.DOUBLE_SEQUENCE_BN.name: DoubleSeqBnConfig,
        # PEFTEnum.PARALLEL_BN.name: ParBnConfig,
        # PEFTEnum.COMPACTER.name: CompacterConfig,
        # PEFTEnum.COMPACTER_PP.name: CompacterPlusPlusConfig,
    },
    "hf": {
        PEFTEnum.LORA.name: LoraConfig,
        PEFTEnum.IA3.name: IA3Config,
        PEFTEnum.LoKr.name: LoKrConfig,
        PEFTEnum.LoHa.name: LoHaConfig,
        PEFTEnum.AdaLoRA.name: AdaLoraConfig,
        PEFTEnum.LlamaAdapter.name: AdaptionPromptConfig,
        PEFTEnum.OFT.name: OFTConfig, # Add Implementation
        PEFTEnum.ReFT.name: NotImplemented

    }
}