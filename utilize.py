import tensorflow as tf
from functools import reduce
from operator import mul

def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    """
    with tf.name_scope(name or "dropout"):
        if is_train is None:
            if keep_prob < 1.0:
                return tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
        else:
            if keep_prob < 1.0:
                out = tf.cond(
                    is_train,
                    lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed),
                    lambda: x
                )
                return out
    """
    with tf.name_scope(name or "dropout"):
        if is_train is None:
            if keep_prob < 1.0:
                return tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
        else:
            if is_train and keep_prob < 1.0:
                return tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            return x

def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
            specified and the `tensor` has a different rank, and exception will be
            thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

def selu(x):
    with tf.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def gelu(x):  # read
    # return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
    input_tensor: float Tensor to perform activation.

    Returns:
    `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))
    return x * cdf

def swish(x):
    return x*tf.nn.sigmoid(x)

def activation_name_to_func(activation_name):
    assert isinstance(activation_name, str)
    if isinstance(activation_name, str):
        if activation_name == 'linear':
            act_fn = tf.identity
        elif activation_name == 'relu':
            act_fn = tf.nn.relu
        elif activation_name == 'elu':
            act_fn = tf.nn.elu
        elif activation_name == 'selu':
            act_fn = selu
        elif activation_name == 'sigmoid':
            act_fn = tf.nn.sigmoid
        elif activation_name == 'tanh':
            act_fn = tf.nn.tanh
        elif activation_name == 'exp':
            act_fn = tf.exp
        elif activation_name == 'log':
            act_fn = tf.log
        elif activation_name == 'gelu':
            act_fn = gelu
        elif activation_name == 'swish':
            act_fn = swish
        elif activation_name == 'lrelu':
            act_fn = tf.nn.leaky_relu
        else:
            raise AttributeError('no activation function named as %s' % activation_name)
    elif hasattr(activation_name, '__call__'):  # callable
        act_fn = activation_name
    else:
        raise AttributeError
    return act_fn

def act_name2fn(afn):
    return activation_name_to_func(afn)

def bn_dense_layer_v2(
        input_tensor, hn, bias, bias_start=0.0, scope=None,
        activation='relu', enable_bn=False,
        wd=0., keep_prob=1.0, is_train=None, dup_num=1, merge_var=False
):
    act_fn = act_name2fn(activation)
    with tf.variable_scope(scope or 'bn_dense_layer'):
        input_tensor = dropout(input_tensor, keep_prob, is_train)
        # the comment use a 3d tensor [bs,sl,hn] as a example
        input_shape = get_shape_list(input_tensor)  # [3]
        assert len(input_shape) >= 2  # at least [bs,hn]
        # merge
        dims_merge = input_shape[:-1]  # [all unrelated dims]
        new_dim = reduce(mul, dims_merge)  # get the merged dim
        new_shape = [new_dim, input_shape[-1]]  # new shape for matmul [2]
        input_tensor_rsp = tf.reshape(input_tensor, new_shape)  #  [xx,dim]

        # dense layer
        input_dim = new_shape[-1]
        if merge_var:
            weight = tf.get_variable('W', shape=[input_dim, hn * dup_num], dtype=tf.float32)
        else:
            weight_list = []
            for i in range(dup_num):
                weight_list.append(tf.get_variable('W_%d' % i, shape=[input_dim, hn]))
            weight = tf.concat(weight_list, -1)
        output_rsp = tf.matmul(input_tensor_rsp, weight)

        if bias:
            if merge_var or dup_num == 1:
                bias_val = tf.get_variable(
                    'bias', shape=[hn * dup_num], dtype=tf.float32,
                    initializer=tf.constant_initializer(bias_start)
                )
            else:
                bias_list = []
                for i in range(dup_num):
                    bias_list.append(
                        tf.get_variable(
                            'bias_%d' % i, shape=[hn], dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_start))
                    )
                bias_val = tf.concat(bias_list, -1)
            output_rsp += bias_val

        # output reshape
        output_shape = dims_merge + [hn * dup_num]  # [3] for [bs,sl,new_hn]
        output = tf.reshape(output_rsp, output_shape)  # [bs,sl,new_hn]

        if enable_bn:
            output = tf.contrib.layers.batch_norm(
                output, center=True, scale=True, is_training=is_train,
                updates_collections=None,  decay=0.9,
                scope='bn')

        if wd:
            tf.add_to_collection('reg_vars', weight)

        return act_fn(output)