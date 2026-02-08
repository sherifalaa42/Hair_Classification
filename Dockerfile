FROM public.ecr.aws/lambda/python:3.10

RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install numpy==1.23.1
RUN pip install pillow


COPY model_2024_hairstyle_v2.tflite .
COPY fun.py .

CMD [ "fun.lambda_handler" ]