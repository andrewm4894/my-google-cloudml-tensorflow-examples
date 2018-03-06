import setuptools

NAME = 'trainer'
VERSION = '1.0'
TENSORFLOW_TRANSFORM = 'tensorflow-transform==0.4'
PROTOBUF = 'protobuf==3.4.0'
TENSORFLOW = 'tensorflow==1.4.0'


if __name__ == '__main__':
  setuptools.setup(name=NAME, version=VERSION, packages=['trainer'],
                   install_requires=[TENSORFLOW_TRANSFORM,PROTOBUF,TENSORFLOW])