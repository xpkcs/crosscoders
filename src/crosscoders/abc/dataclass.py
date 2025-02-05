

from dataclasses import asdict, dataclass

from crosscoders.utils import dataclass_repr




@dataclass
class DataclassABC:

    def __repr__(self) -> str:
        
        return dataclass_repr(self)


    def asdict(self):

        return asdict(self)