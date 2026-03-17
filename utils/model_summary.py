import torch.nn as nn
import torch
import numpy as np

'''
---- 1) FLOPs: floating point operations
---- 2) #Activations: the number of elements of all 'Conv2d' outputs
---- 3) #Conv2d: the number of 'Conv2d' layers
'''

def get_model_flops(model, input_res, print_per_layer_stat=True,
                              input_constructor=None):
    assert type(input_res) is tuple, 'Please provide the size of the input image.'
    assert len(input_res) >= 3, 'Input image should have 3 dimensions.'
    flops_model = add_flops_counting_methods(model)
    flops_model.eval().start_flops_count()
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        device = list(flops_model.parameters())[-1].device
        batch = torch.FloatTensor(1, *input_res).to(device)
        _ = flops_model(batch)

    if print_per_layer_stat:
        print_model_with_flops(flops_model)
    flops_count = flops_model.compute_average_flops_cost()
    flops_model.stop_flops_count()

    return flops_count

def get_model_activation(model, input_res, input_constructor=None):
    assert type(input_res) is tuple, 'Please provide the size of the input image.'
    assert len(input_res) >= 3, 'Input image should have 3 dimensions.'
    activation_model = add_activation_counting_methods(model)
    activation_model.eval().start_activation_count()
    if input_constructor:
        input = input_constructor(input_res)
        _ = activation_model(**input)
    else:
        device = list(activation_model.parameters())[-1].device
        batch = torch.FloatTensor(1, *input_res).to(device)
        _ = activation_model(batch)

    activation_count, num_conv = activation_model.compute_average_activation_cost()
    activation_model.stop_activation_count()

    return activation_count, num_conv


def get_model_complexity_info(model, input_res, print_per_layer_stat=True, as_strings=True,
                              input_constructor=None):
    assert type(input_res) is tuple
    assert len(input_res) >= 3
    flops_model = add_flops_counting_methods(model)
    flops_model.eval().start_flops_count()
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        batch = torch.FloatTensor(1, *input_res)
        _ = flops_model(batch)

    if print_per_layer_stat:
        print_model_with_flops(flops_model)
    flops_count = flops_model.compute_average_flops_cost()
    params_count = get_model_parameters_number(flops_model)
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num):
    if params_num // 10 ** 6 > 0:
        return str(round(params_num / 10 ** 6, 2)) + ' M'
    elif params_num // 10 ** 3:
        return str(round(params_num / 10 ** 3, 2)) + ' k'
    else:
        return str(params_num)


def print_model_with_flops(model, units='GMac', precision=3):
    total_flops = model.compute_average_flops_cost()

    def accumulate_flops(self):
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self):
        accumulated_flops_cost = self.accumulate_flops()
        return ', '.join([flops_to_string(accumulated_flops_cost, units=units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(net_main_module):
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()
    return net_main_module


def compute_average_flops_cost(self):
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__
    return flops_sum


def start_flops_count(self):
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    self.apply(add_flops_counter_variable_or_reset)


def add_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return

        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            handle = module.register_forward_hook(conv_flops_counter_hook)
        elif isinstance(module, (nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
            handle = module.register_forward_hook(relu_flops_counter_hook)
        elif isinstance(module, nn.Linear):
            handle = module.register_forward_hook(linear_flops_counter_hook)
        elif isinstance(module, (nn.BatchNorm2d)):
            handle = module.register_forward_hook(bn_flops_counter_hook)
        else:
            handle = module.register_forward_hook(empty_flops_counter_hook)
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        module.__flops__ = 0


# ---- Internal functions
def is_supported_instance(module):
    if isinstance(module, (
        nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d,
        nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6,
        nn.Linear,
        nn.BatchNorm2d,
    )):
        return True
    return False


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    if hasattr(conv_module, '__mask__'):
        # (b, 1, ...)
        output_height, output_width = output.shape[2:]
        flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height, output_width)
        active_elements_count = flops_mask.sum()

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    output_last_dim = output.shape[-1]
    bias_flops = output_last_dim if module.bias is not None else 0
    module.__flops__ += int(np.prod(input.shape) * output_last_dim + bias_flops)


def bn_flops_counter_hook(module, input, output):
    input = input[0]
    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


# ---- Count the number of convolutional layers and the activation
def add_activation_counting_methods(net_main_module):
    net_main_module.start_activation_count = start_activation_count.__get__(net_main_module)
    net_main_module.stop_activation_count = stop_activation_count.__get__(net_main_module)
    net_main_module.reset_activation_count = reset_activation_count.__get__(net_main_module)
    net_main_module.compute_average_activation_cost = compute_average_activation_cost.__get__(net_main_module)

    net_main_module.reset_activation_count()
    return net_main_module


def compute_average_activation_cost(self):
    activation_sum = 0
    num_conv = 0
    for module in self.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            activation_sum += module.__activation__
            num_conv += 1
    return activation_sum, num_conv


def start_activation_count(self):
    self.apply(add_activation_counter_hook_function)


def stop_activation_count(self):
    self.apply(remove_activation_counter_hook_function)


def reset_activation_count(self):
    self.apply(add_activation_counter_variable_or_reset)


def add_activation_counter_hook_function(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        if hasattr(module, '__activation_handle__'):
            return
        handle = module.register_forward_hook(conv_activation_counter_hook)
        module.__activation_handle__ = handle


def remove_activation_counter_hook_function(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        if hasattr(module, '__activation_handle__'):
            module.__activation_handle__.remove()
            del module.__activation_handle__


def add_activation_counter_variable_or_reset(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        module.__activation__ = 0


def conv_activation_counter_hook(conv_module, input, output):
    active_elements_count = output.numel()
    conv_module.__activation__ += int(active_elements_count)
