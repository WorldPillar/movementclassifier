import json


class Config:
    __default = {'csv_path': 'csv_data', 'model_path': 'models', 'model_name': 'new_model.joblib', 'names': {}}
    csv_path = 'csv_data'
    model_path = 'models'
    model_name = 'new_model.joblib'
    names = {}

    def __init__(self, data: dict):
        self.csv_path = data.get('csv_path', self.__default['csv_path'])
        self.model_path = data.get('model_path', self.__default['model_path'])
        self.model_name = data.get('model_name', self.__default['model_name'])

    @staticmethod
    def set_names(names):
        Config.names = names
        Config.__save_config()

    @staticmethod
    def get_model():
        return f'{Config.model_path}/{Config.model_name}'

    @staticmethod
    def __save_config():
        json_data = {'csv_path': Config.csv_path, 'model_path': Config.model_path,
                     'model_name': Config.model_name, 'names': Config.names}
        with open('config.json', 'w') as f:
            json.dump(json_data, f)
        pass

    @staticmethod
    def load_config():
        data = Config.__default
        try:
            f = open('config.json', 'r')
            data = json.load(f)
            f.close()
        except FileNotFoundError:
            data = Config.__default
        finally:
            Config.csv_path = data.get('csv_path', Config.__default['csv_path'])
            Config.model_path = data.get('model_path', Config.__default['model_path'])
            Config.model_name = data.get('model_name', Config.__default['model_name'])
            Config.names = data.get('names', Config.__default['names'])

        Config.__save_config()
