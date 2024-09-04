from enum import Enum

from adapters import LoRAConfig, IA3Config, SeqBnConfig, DoubleSeqBnConfig, ParBnConfig, CompacterConfig, \
    CompacterPlusPlusConfig


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


pefts_configuration = {
    PEFTEnum.BITFIT.name: NotImplemented, # see how to implement
    PEFTEnum.LORA.name: LoRAConfig,
    PEFTEnum.IA3.name: IA3Config,
    PEFTEnum.LO_REFT.name: NotImplemented, # ReftConfig, # use pyreft
    PEFTEnum.NO_REFT.name: NotImplemented,
    PEFTEnum.DI_REFT.name: NotImplemented,
    PEFTEnum.SEQUENCE_BN.name: SeqBnConfig,
    PEFTEnum.DOUBLE_SEQUENCE_BN.name: DoubleSeqBnConfig,
    PEFTEnum.PARALLEL_BN.name: ParBnConfig,
    PEFTEnum.COMPACTER.name: CompacterConfig,
    PEFTEnum.COMPACTER_PP.name: CompacterPlusPlusConfig
}
