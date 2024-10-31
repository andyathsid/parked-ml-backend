FROM tensorflow/serving:2.14.0

COPY model_hw_newhandpd_aug-ilum_resnet50 /models/model_hw_newhandpd_aug-ilum_resnet50/1
ENV MODEL_NAME="model_hw_newhandpd_aug-ilum_resnet50"
