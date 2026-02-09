# Use the official Ubuntu base image
FROM ubuntu:jammy

# Set the working directory
WORKDIR /qec

RUN mkdir -p /LLI

COPY requirements.txt ./

# Python packages
RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.10 python3-distutils python3-pip python3-apt

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

# Other SW
RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y git vim jq bc make automake cmake libnuma-dev \
    && apt-get install -y rsync htop curl build-essential parallel \
    && apt-get install -y pkg-config libffi-dev libgmp-dev \
    && apt-get install -y libssl-dev libtinfo-dev libsystemd-dev \
    && apt-get install -y zlib1g-dev make g++ wget libncursesw5 libtool autoconf \
    && apt-get clean

# Install ghcup
ENV BOOTSTRAP_HASKELL_NONINTERACTIVE=1
RUN bash -c "curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh"
RUN bash -c "curl -sSL https://get.haskellstack.org/ | sh"

# Add ghcup to PATH
ENV PATH=${PATH}:/root/.local/bin
ENV PATH=${PATH}:/root/.ghcup/bin

# Install cabal
RUN bash -c "ghcup upgrade"
RUN bash -c "ghcup install cabal 3.10.2.1"
RUN bash -c "ghcup set cabal 3.10.2.1"

# Install GHC
RUN bash -c "ghcup install ghc 9.4.4"
RUN bash -c "ghcup set ghc 9.4.4"

# Update Path to include Cabal and GHC exports
RUN bash -c "echo PATH="$HOME/.local/bin:$PATH" >> $HOME/.bashrc"
RUN bash -c "echo export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH" >> $HOME/.bashrc"
RUN bash -c "source $HOME/.bashrc"

# Update cabal
RUN bash -c "cabal update"

# Set LD path for lsqecc build
RUN bash -c "echo export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/qec/liblsqecc/external/rotation-decomposer/newsynth/" >> $HOME/.bashrc"


