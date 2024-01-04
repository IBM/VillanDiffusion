import argparse
from dataclasses import dataclass, asdict, is_dataclass
import json
import os
from typing import Dict, List, Union, Tuple

@dataclass
class DummyClass:
    pass

def yield_default(config: Union[dataclass, dict], return_dict: bool=False) -> dataclass:
    config_dict: dict = {}
    res_dataclass: dataclass = DummyClass()
    
    if is_dataclass(config):
        config_dict = asdict(config)
    for key, val in config_dict.items():
        setattr(res_dataclass, key, val)
        
    if return_dict:
        return asdict(res_dataclass)
    return res_dataclass

@dataclass
class ArgParser:
    ARGS_EXPORT: str = 'export'
    ARGS_DEFAULT: str = 'default'
    ARGS_CHOICES: str = 'choices'
    ARGS_HELP: str = 'help'
    RECEIVE_ARGS_MODE_DEFAULT: str = 'AS_DEFAULT'
    RECEIVE_ARGS_MODE_CONFIG: str = 'AS_CONFIG'
    def __init__(self, config_key: str=None, config: Union[dict, dataclass]=None, file_name=None):
        # self.__config: dataclass = config
        self.__config_dict: dict = {}
        self.__file_name: str = file_name
        if (config_key is not None) and (config is not None):
            self.add(config_key=config_key, config=config)
    
    @staticmethod
    def __rm_keys_from_dict(d: dict, keys: Union[List[object], object]):
        if isinstance(keys, (list, tuple)):
            for key in keys:
                del d[key]
        else:
            del d[key]
        return d
        
    def __add_args(self, parser: argparse.ArgumentParser, key: str, val: dict, mode: str):
        if mode == ArgParser.RECEIVE_ARGS_MODE_CONFIG:
            print(f"Add arguement: --{key}: {val}")
            # rm_val: dict = ArgParser.__rm_keys_from_dict(d=val, keys=[ArgParser.ARGS_HELP])
            # if ArgParser.ARGS_HELP in val:
            #     parser.add_argument(f'--{key}', help=val[ArgParser.ARGS_HELP], **rm_val)
            parser.add_argument(f'--{key}', **val)
        elif mode == ArgParser.RECEIVE_ARGS_MODE_DEFAULT:
            print(f"Add arguement: --{key}: default: {val}")
            parser.add_argument(f'--{key}', default=val)
        else:
            raise NotImplementedError()
        return parser
    
    @staticmethod
    def __default_help_choice(config: dict):
        if ArgParser.ARGS_CHOICES in config:
            config[ ArgParser.ARGS_HELP] += f", choice: {str(config[ArgParser.ARGS_CHOICES])}"
        return config
    
    @staticmethod
    def __default_help_default(config: dict):
        if ArgParser.ARGS_DEFAULT in config:
            config[ArgParser.ARGS_HELP] += f", default: {str(config[ArgParser.ARGS_DEFAULT])}"
        return config

    def receive_args(self, config_key: str, mode: str=RECEIVE_ARGS_MODE_CONFIG, help_choice: bool=True, help_default: bool=True, *args, **kwargs):
        config_dict = self.__config_dict[config_key]
        
        default_vals: dict = {}
        parser = argparse.ArgumentParser(args, kwargs)
        # print(f"config_dict: {config_dict}")
        for key, val in config_dict.items():
            # print(f"key: {key}, val: {val}")
            if mode == ArgParser.RECEIVE_ARGS_MODE_CONFIG:
                if isinstance(val, dict) and ArgParser.ARGS_EXPORT in val:
                    if val[ArgParser.ARGS_EXPORT]:
                        del val[ArgParser.ARGS_EXPORT]
                        # parser.add_argument('--test_tmp', default='test_tmp')
                        val = self.__default_help_default(config=val)
                        val = self.__default_help_choice(config=val)
                        parser = self.__add_args(parser=parser, key=key, val=val, mode=ArgParser.RECEIVE_ARGS_MODE_CONFIG)
                    else:
                        default_vals[key] = val[ArgParser.ARGS_DEFAULT]
                else:
                    default_vals[key] = val
            elif mode == ArgParser.RECEIVE_ARGS_MODE_DEFAULT:
                default_vals[key] = val
        
        parse_args = parser.parse_args()
        if not isinstance(parse_args, dict):
            parse_args: dict = parse_args.__dict__
        updated_config: dict = ArgParser.default_update_rule(default_vals, parse_args)
        # print(f"-> default_vals: {default_vals}")
        # print(f"-> parse_args: {parse_args}")
        # print(f"-> updated_config: {updated_config}")
        self.__config_dict[config_key] = updated_config
        
        return self
    
    def load(self, file_name: str=None, config_key: str=None, not_exist_ok: bool=True):
        if file_name is None:
            file_name = self.__file_name
            
        try:
            with open(file_name, 'r') as f:
                data: dict = json.load(f)
        except FileNotFoundError as file_not_found:
            if not_exist_ok:
                data: dict = {}
            else:
                raise FileNotFoundError(f"file: {file_name} not found") from file_not_found
        
        if config_key is None:
            config_key = file_name
            
        self.__config_dict[config_key] = data
        
        return self
    
    def save(self, file_name: str=None, config_key: str=None, overwrite: bool=True):
        if os.path.isfile(file_name) and not overwrite:
           return self 
       
        if file_name is None:
            file_name = self.__file_name
            
        if config_key is None:
            config_key = file_name
            
        data = json.dumps(self.__config_dict[config_key], indent=4)
        with open(file_name, "w") as f:
            f.write(data)
            
        return self
    
    def add(self, config: Union[dict, dataclass], config_key: str):
        if isinstance(config, dict):
            self.__config_dict[config_key] = config
        elif is_dataclass(config):
            self.__config_dict[config_key] = asdict(config)
        else:
            raise NotImplementedError()
        return self
    
    def update(self, in_config_keys: Union[List[str], Tuple[str]], out_config_keys: Union[List[str], Tuple[str], str], update_rule: callable=None):
        if not isinstance(in_config_keys, (list, tuple)):
            raise TypeError
        
        if update_rule is None:
            res = ArgParser.default_update_rule(*[self.__config_dict[key] for key in in_config_keys])
        else:
            res = update_rule(*[self.__config_dict[key] for key in in_config_keys])
        
        if isinstance(out_config_keys, (list, tuple)):
            if len(out_config_keys) == 1:
                self.__config_dict[out_config_keys[0]] = res
                
            if len(out_config_keys) != len(res):
                raise ValueError
            for itm, key in zip(itm, out_config_keys):
                self.__config_dict[key] = itm
        else:
            self.__config_dict[out_config_keys] = res
        
        return self
        
    @staticmethod
    def default_update_rule(*config_ls):
        if len(config_ls) == 0:
            raise ValueError("Empty config list, nothing to update")
        if len(config_ls) <= 1:
            return config_ls[0]
        
        res = config_ls[0]
        for config in config_ls[1:]:
            for key, val in config.items():
                if val is not None:
                    res[key] = val
        return res
        
    def parse(self, config_key: str, dataclass_type: callable=None, return_dict: bool=False) -> dataclass:
        if config_key is None:
            raise ValueError("Arguement config shouldn't be None")
        if dataclass_type is None and not return_dict:
            raise ValueError("Arguement dataclass_type is required for returning the dataclass")
        if config_key not in self.__config_dict:
            raise ValueError(f"config_key should be add before, added config_key: {self.__config_dict.keys()}")
        
        res = dataclass_type()
        for key, val in self.__config_dict[config_key].items():
            setattr(res, key, val)
        return res