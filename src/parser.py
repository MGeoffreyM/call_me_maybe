import json
from enum import Enum
from pydantic import BaseModel, Field


class types(Enum):
    INT = 'number'
    STR = 'string'
    BOOL = 'true' or 'false'


class params(Enum):
    NAME = 'name'
    DESCRIPTION = 'description'
    PARAMETER = 'parameters'
    RETURN = 'returns'
    TYPE = 'type'


class Fonction(BaseModel):
    name: str = ''
    description: str = ''
    args: dict[str, type] = Field(default_factory=dict)
    result: dict[str, type] = Field(default_factory=dict)


class Parser(BaseModel):
    list_fonction: list[Fonction] = Field(default_factory=list)

    def read_file(self, file: str) -> None:
        with open(file, 'r') as f:
            data = json.load(f)
        # return data
        for values in data:
            nm = ''
            desc = ''
            arg = {}
            res = {}
            for key, value in values.items():
                if key == params.NAME:
                    nm = value.strip('\"')
                elif key == params.DESCRIPTION:
                    desc = value.strip('\"')
                elif key == params.PARAMETER:
                    for a, b in value.items():
                        _, c = b.split(':').strip('\" ')
                        if b == types.INT:
                            arg[a] = int
                        elif b == types.STR:
                            arg[a] = str
                        elif b == types.BOOL:
                            arg[a] = bool
                elif key == params.RETURN:
                    for a, b in value.items():
                        res[a] = b.split(':', 1).strip('\" ')
            self.list_fonction.append(Fonction(name=nm,
                                               description=desc,
                                               args=arg,
                                               result=res))

