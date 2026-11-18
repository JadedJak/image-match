FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1
# CLIP weights are NOT baked into the image (see docker-compose.yml's
# hf-cache volume + README "CLIP weight cache" instead) -- they live in a
# plain host folder, bind-mounted in at runtime. Earlier this project
# tried baking them into an image layer, then into a BuildKit cache mount;
# both are tied to Docker's build machinery and both ended up causing a
# redundant multi-GB re-download on a metered connection when that
# machinery didn't behave the way it looked like it would. A host folder
# is the one thing that can't be silently invalidated by a Dockerfile
# edit, a cache eviction, or anything else Docker-related -- it just sits
# on disk until you delete it yourself. HF_HUB_OFFLINE=1 is set
# unconditionally, so a normal container run can never silently fall back
# to a network fetch even if the mount is somehow empty -- it fails loudly
# instead (see README for the one-time cache-warming command that's the
# only path allowed to actually download).
ENV HF_HOME=/opt/hf-cache
ENV HF_HUB_OFFLINE=1
EXPOSE 7860

CMD ["python", "-m", "app.main"]
