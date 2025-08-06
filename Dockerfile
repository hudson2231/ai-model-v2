FROM r8.im/cog-base:cuda11.8-python3.10-torch2.1.0

RUN pip install https://github.com/replicate/cog-runtime/releases/download/v0.1.0-beta5/coglet-0.1.0b5-py3-none-any.whl

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.txt

WORKDIR /src
COPY . /src
