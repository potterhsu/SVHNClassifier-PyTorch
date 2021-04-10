# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: svhn_classifier.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='svhn_classifier.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x15svhn_classifier.proto\"*\n\x05Image\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\x13\n\x0b\x63\x65nter_crop\x18\x02 \x01(\x08\",\n\nPrediction\x12\x0e\n\x06length\x18\x01 \x01(\x05\x12\x0e\n\x06\x64igits\x18\x02 \x03(\x05\x32\x32\n\x0eSvhnClassifier\x12 \n\x07Predict\x12\x06.Image\x1a\x0b.Prediction\"\x00\x62\x06proto3')
)




_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='Image.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='center_crop', full_name='Image.center_crop', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=25,
  serialized_end=67,
)


_PREDICTION = _descriptor.Descriptor(
  name='Prediction',
  full_name='Prediction',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='length', full_name='Prediction.length', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='digits', full_name='Prediction.digits', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=69,
  serialized_end=113,
)

DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
DESCRIPTOR.message_types_by_name['Prediction'] = _PREDICTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), dict(
  DESCRIPTOR = _IMAGE,
  __module__ = 'svhn_classifier_pb2'
  # @@protoc_insertion_point(class_scope:Image)
  ))
_sym_db.RegisterMessage(Image)

Prediction = _reflection.GeneratedProtocolMessageType('Prediction', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTION,
  __module__ = 'svhn_classifier_pb2'
  # @@protoc_insertion_point(class_scope:Prediction)
  ))
_sym_db.RegisterMessage(Prediction)



_SVHNCLASSIFIER = _descriptor.ServiceDescriptor(
  name='SvhnClassifier',
  full_name='SvhnClassifier',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=115,
  serialized_end=165,
  methods=[
  _descriptor.MethodDescriptor(
    name='Predict',
    full_name='SvhnClassifier.Predict',
    index=0,
    containing_service=None,
    input_type=_IMAGE,
    output_type=_PREDICTION,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_SVHNCLASSIFIER)

DESCRIPTOR.services_by_name['SvhnClassifier'] = _SVHNCLASSIFIER

# @@protoc_insertion_point(module_scope)
