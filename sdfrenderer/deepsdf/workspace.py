#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
# Source: https://github.com/facebookresearch/DeepSDF

import json
import os
import torch
from torch import nn

model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"
latent_codes_subdir = "LatentCodes"
logs_filename = "Logs.pth"
reconstructions_subdir = "Reconstructions"
reconstruction_meshes_subdir = "Meshes"
reconstruction_codes_subdir = "Codes"
specifications_filename = "specs.json"
data_source_map_filename = ".datasources.json"
evaluation_subdir = "Evaluation"
sdf_samples_subdir = "SdfSamples"
surface_samples_subdir = "SurfaceSamples"
normalization_param_subdir = "NormalizationParameters"


def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file " +
            '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))


def load_model_parameters(experiment_directory, checkpoint, decoder):

    filename = os.path.join(experiment_directory, model_params_subdir, checkpoint + ".pth")

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename)

    decoder.load_state_dict(data["model_state_dict"])

    return data["epoch"]


def build_decoder(experiment_directory, experiment_specs):

    arch = __import__("networks." + experiment_specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = experiment_specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **experiment_specs["NetworkSpecs"]).cuda()

    return decoder


def load_decoder(experiment_directory, experiment_specs, checkpoint, data_parallel=True):

    decoder = build_decoder(experiment_directory, experiment_specs)

    if data_parallel:
        decoder = torch.nn.DataParallel(decoder)

    epoch = load_model_parameters(experiment_directory, checkpoint, decoder)

    return (decoder, epoch)


def load_latent_vectors(experiment_directory, checkpoint):

    filename = os.path.join(experiment_directory, latent_codes_subdir, checkpoint + ".pth")

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include a latent code file " +
            'for checkpoint "{}"'.format(experiment_directory, checkpoint)
        )

    data = torch.load(filename)

    num_vecs = data["latent_codes"].size()[0]
    latent_size = data["latent_codes"].size()[2]

    lat_vecs = []
    for i in range(num_vecs):
        lat_vecs.append(data["latent_codes"][i].cuda())

    return lat_vecs


def get_data_source_map_filename(data_dir):
    return os.path.join(data_dir, data_source_map_filename)


def get_reconstructed_mesh_filename(experiment_dir, epoch, dataset, class_name, instance_name):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        class_name,
        instance_name + ".ply",
    )


def get_reconstructed_code_filename(experiment_dir, epoch, dataset, class_name, instance_name):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        class_name,
        instance_name + ".pth",
    )


def get_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def setup_dsdf(dir, mode='eval', precision=torch.float16):
    specs_filename = os.path.splitext(dir)[0] + '.json'
    if not os.path.isfile(specs_filename):
        raise Exception('The experiment directory does not include specifications file "specs.json"')
    specs = json.load(open(specs_filename))
    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    specs["NetworkSpecs"].pop('samples_per_scene', None)  # remove samples_per_scene to get scale for a single model
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(dir)
    saved_model_epoch = saved_model_state["epoch"]
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module
    convert_to_precision(decoder, precision)

    if mode == 'train':
        decoder.train()
    elif mode == 'eval':
        decoder.eval()

    return decoder, latent_size


def convert_to_precision(model, precision):
    model.to(dtype=precision)
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()
