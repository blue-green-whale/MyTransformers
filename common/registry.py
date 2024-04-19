import os
class Regitry:
    mapping = {
        "model_mapping":{},
        "pipeline_model_mapping":{},
        "train_model_mapping":{},
        "model_config_mapping":{},
        "dataset_mapping":{},
        "info_manager_mapping":{},
        "tokenizer_mapping":{},
        "paths_mapping":{}
    }

    @classmethod
    def register_path(cls, name, path):
        if name in cls.mapping['paths_mapping']:
            raise KeyError(
                "Name '{}' already registered for {}.".format(
                    name, cls.mapping["paths_mapping"][name]
                )
            )
        cls.mapping['paths_mapping'][name] = path


    @classmethod
    def register_dataset(cls, name):

        def warp(dataset_cls):
            if name in cls.mapping['dataset_mapping']；
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["dataset_mapping"][name]
                    )
                )            
            cls.mapping['dataset_mapping'][name] = dataset_cls
            return func
        return wrap

    @classmethod
    def register_info_manager(cls, name):

        def wrap(func):
            if name in cls.mapping['info_manager_mapping']:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["info_manager_mapping"][name]
                    )
                )
            cls.mapping['info_manager_mapping'][name] = func
            return func
        return wrap

    @classmethod
    def register_model(cls, name):
        def wrap(model_cls):
            if model_cls in cls.mapping['model_mapping']:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_mapping"][name]
                    )
                )
            cls.mapping['model_mapping'][name] = model_cls
            return model_cls
        return wrap
    
    @classmethod
    def register_pipeline_model(cls, name):
        def wrap(model_cls):
            if model_cls in cls.mapping['pipeline_model_mapping']:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["pipeline_model_mapping"][name]
                    )
                )
            cls.mapping['pipeline_model_mapping'][name] = model_cls
            return model_cls
        return wrap

    @classmethod
    def register_model_config(cls, name):

        def wrap(model_cls):
            if model_cls in cls.mapping['model_mapping']:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_config_mapping"][name]
                    )
                )
            cls.mapping['model_config_mapping'][name] = model_cls
            return model_cls
        return wrap

    @classmethod
    def register_train_model(cls, name):
        from model.base_model import BaseModel
        def wrap(model_cls):
            assert(issubclass(model_cls, BaseModel))
            if model_cls in cls.mapping['model_mapping']:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["train_model_mapping"][name]
                    )
                )
            cls.mapping['train_model_mapping'][name] = model_cls
            return model_cls
        return wrap

    @classmethod
    def register_info_manager(cls, name):

        def wrap(func):
            if name in cls.mapping['info_manager_mapping']:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["info_manager_mapping"][name]
                    )
                )
            cls.mapping['info_manager_mapping'][name] = func
            return func
        return wrap
    
    @classmethod
    def register_tokenizer(cls, name):

        def wrap(tokenizer_cls):
            if name in cls.mapping['tokenizer_mapping']:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["tokenizer_mapping"][name]
                    )
                )
            cls.mapping['tokenizer_mapping'][name] = tokenizer_cls
            return tokenizer_cls
        return wrap
    
    @classmethod
    def get_model_class(cls, name):
        result = cls.mapping["model_mapping"].get(name, None)
        if result is None:
            raise ValueError(f"Can not find name: {name} in model mapping, \
supported models are listed below:{cls.list_models()}")
        else:
            return result
    
    @classmethod
    def get_pipeline_model_class(cls, name):
        result = cls.mapping["pipeline_model_mapping"].get(name, None)
        if result is None:
            raise ValueError(f"Can not find name: {name} in pipeline model mapping, \
supported pipeline models are listed below:{cls.list_pipeline_models()}")
        else:
            return result
    
    @classmethod
    def get_model_config_class(cls, name):
        result = cls.mapping["model_config_mapping"].get(name, None)
        if result is None:
            raise ValueError(f"Can not find name: {name} in model config mapping, \
supported model configs are listed below:{cls.list_model_configs()}")
        else:
            return result
    
    @classmethod
    def get_train_model_class(cls, name):
        result = cls.mapping["train_model_mapping"].get(name, None)
        if result is None:
            raise ValueError(f"Can not find name: {name} in train model mapping, \
supported train models are listed below:{cls.list_train_models()}")
        else:
            return result
    
    @classmethod
    def get_tokenizer_class(cls, name):
        result =  cls.mapping["tokenizer_mapping"].get(name, None)
        if result is None:
            raise ValueError(f"Can not find name:{name} in tokenizer mapping, \
supported tokenizer are listed below:{cls.list_tokenizers()}")
        else:
            return result

    @classmethod
    def get_dataset_class(cls, name):
        result =  cls.mapping["dataset_mapping"].get(name, None)
        if result is None:
            raise ValueError(f"Can not find name:{name} in dataset mapping, \
supported dataset are listed below:{cls.list_datasets()}")
        else:
            return result
    
    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths_mapping"].get(name, None)
    
    @classmethod
    def get_paths(cls, args):
        # by doing this you are not need to provide paths in your script
        paths_mapping = cls.mapping["paths_mapping"]
        for k,v in cls.mapping["paths_mapping"].items():
            if not os.path.isfile(v):
                paths_mapping[k] = None
        tokenizer_name = "tokenizer_" + args.model_name
        model_name = "model_"  + '_'.join([args.model_name, args.variant])
        dataset_name = "dataset_" + args.dataset_name
        args.tokenizer_path = args.tokenizer_path if args.tokenizer_path else paths_mapping.get(tokenizer_name, None)
        args.dataset_path = args.dataset_path if args.dataset_path else paths_mapping.get(dataset_name, None)
        args.ckpt_path = args.ckpt_path if args.ckpt_path else paths_mapping.get(model_name, None)
        return args
    
    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_mapping"].keys())
    
    @classmethod
    def list_pipeline_models(cls):
        return sorted(cls.mapping["pipeline_model_mapping"].keys())

    @classmethod
    def list_model_configs(cls):
        return sorted(cls.mapping["model_config_mapping"].keys())
    
    @classmethod
    def list_train_models(cls):
        return sorted(cls.mapping["train_model_mapping"].keys())

    @classmethod
    def list_paths(cls):
        return sorted(cls.mapping["paths_mapping"].keys())
    
    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping["datasets_mapping"].keys())
    
    @classmethod
    def list_info_managers(cls):
        return sorted(cls.mapping["info_manager_mapping"].keys())
    
    @classmethod
    def list_tokenizers(cls):
        return sorted(cls.mapping["tokenizer_mapping"].keys())

    @classmethod
    def list_all(cls):
        return {k:sorted(v.keys()) for k,v in cls.mapping.items()}

    
registry = Regitry()