from decouple import config

class Config():
    FLASK_APP = config('FLASK_APP')
    FLASK_ENV = config('FLASK_ENV')


class DevelopmentConfig(Config):
    DEBUG = True


config = {
    'development': DevelopmentConfig
}