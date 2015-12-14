"""
Costs for use with the MLP model class.
"""
__authors__ = 'Vincent Archambault-Bouffard, Ian Goodfellow'
__copyright__ = "Copyright 2013, Universite de Montreal"

from functools import wraps
import operator
import warnings

import theano
from theano import tensor as T
from theano.compat.six.moves import reduce

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin, NullDataSpecsMixin
from pylearn2.utils import safe_izip
from pylearn2.utils.exc import reraise_as


class Default(DefaultDataSpecsMixin, Cost):
    """The default Cost to use with an MLP.

    It simply calls the MLP's cost_from_X method.
    """

    supervised = True

    def expr(self, model, data, **kwargs):
        """Returns a theano expression for the cost function.

        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        return model.cost_from_X(data)

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False


class WeightDecay(NullDataSpecsMixin, Cost):
    """L2 regularization cost for MLP.

    coeff * sum(sqr(weights)) for each set of weights.

    Parameters
    ----------
    coeffs : dict
        Dictionary with layer names as its keys,
        specifying the coefficient to multiply
        with the cost defined by the squared L2 norm of the weights for
        each layer.

        Each element may in turn be a list, e.g., for CompositeLayers.
    """

    def __init__(self, coeffs):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, **kwargs):
        """Returns a theano expression for the cost function.

        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        total_cost : theano.gof.Variable
            coeff * sum(sqr(weights))
            added up for each set of weights.
        """
        self.get_data_specs(model)[0].validate(data)
        assert T.scalar() != 0.  # make sure theano semantics do what I want

        def wrapped_layer_cost(layer, coeff):
            try:
                return layer.get_weight_decay(coeff)
            except NotImplementedError:
                if coeff == 0.:
                    return 0.
                else:
                    reraise_as(NotImplementedError(str(type(
                        layer)) + " does not implement "
                                                   "get_weight_decay."))

        if isinstance(self.coeffs, list):
            warnings.warn("Coefficients should be given as a dictionary "
                          "with layer names as key. The support of "
                          "coefficients as list would be deprecated from "
                          "03/06/2015")
            layer_costs = [
                wrapped_layer_cost(layer, coeff)
                for layer, coeff in safe_izip(model.layers, self.coeffs)
            ]
            layer_costs = [cost for cost in layer_costs if cost != 0.]
        else:
            layer_costs = []
            for layer in model.layers:
                layer_name = layer.layer_name
                if layer_name in self.coeffs:
                    cost = wrapped_layer_cost(layer, self.coeffs[layer_name])
                    if cost != 0.:
                        layer_costs.append(cost)

        if len(layer_costs) == 0:
            rval = T.as_tensor_variable(0.)
            rval.name = '0_weight_decay'
            return rval
        else:
            total_cost = reduce(operator.add, layer_costs)
        total_cost.name = 'MLP_WeightDecay'

        assert total_cost.ndim == 0

        total_cost.name = 'weight_decay'

        return total_cost

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False


class L1WeightDecay(NullDataSpecsMixin, Cost):
    """L1 regularization cost for MLP.

    coeff * sum(abs(weights)) for each set of weights.

    Parameters
    ----------
    coeffs : dict
        Dictionary with layer names as its keys,
        specifying the coefficient to multiply
        with the cost defined by the squared L2 norm of the weights for
        each layer.

        Each element may in turn be a list, e.g., for CompositeLayers.
    """

    def __init__(self, coeffs):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, **kwargs):
        """Returns a theano expression for the cost function.

        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        total_cost : theano.gof.Variable
            coeff * sum(abs(weights))
            added up for each set of weights.
        """

        assert T.scalar() != 0.  # make sure theano semantics do what I want
        self.get_data_specs(model)[0].validate(data)
        if isinstance(self.coeffs, list):
            warnings.warn("Coefficients should be given as a dictionary "
                          "with layer names as key. The support of "
                          "coefficients as list would be deprecated "
                          "from 03/06/2015")
            layer_costs = [
                layer.get_l1_weight_decay(coeff)
                for layer, coeff in safe_izip(model.layers, self.coeffs)
            ]
            layer_costs = [cost for cost in layer_costs if cost != 0.]

        else:
            layer_costs = []
            for layer in model.layers:
                layer_name = layer.layer_name
                if layer_name in self.coeffs:
                    cost = layer.get_l1_weight_decay(self.coeffs[layer_name])
                    if cost != 0.:
                        layer_costs.append(cost)

        if len(layer_costs) == 0:
            rval = T.constant(0., dtype=theano.config.floatX)
            rval.name = '0_l1_penalty'
            return rval
        else:
            total_cost = reduce(operator.add, layer_costs)
        total_cost.name = 'MLP_L1Penalty'

        assert total_cost.ndim == 0

        total_cost.name = 'l1_penalty'

        return total_cost

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False


class FusedLasso(NullDataSpecsMixin, Cost):
    """Fused Lasso regularization cost for MLP.

    coeff * sum(abs(diff(weights))) for each set of weights.

    Parameters
    ----------
    coeffs : dict
        Dictionary with layer names as its keys,
        specifying the coefficient to multiply
        with the cost defined by the fused lasso norm of the weights for
        each layer.

        Each element may in turn be a list, e.g., for CompositeLayers.
    """

    @staticmethod
    def _diff_operator4D(W, axis):
        _, _, wrows, wcols = W.get_value().shape

        def construct_d(dim):
            """Costructs the finite difference matrix.

            Params:
                dim: the output dimension, in rows

            Output:
                the finite difference matrix, with shape = (dim, dim-1)
            """
            import scipy.linalg
            import numpy as np
            firstcol = np.zeros(dim, dtype=theano.config.floatX)
            firstcol[0] = -1.
            firstcol[1] = 1.
            firstrow = np.zeros(dim, dtype=theano.config.floatX)
            firstrow[0] = -1.
            return scipy.linalg.toeplitz(firstcol, firstrow)[:, :dim - 1]

        def fn(Wt, D):
            """The function to be passed to theano.map.

            The order of the parameters is fixed by scan: the output of the
            prior call to fn (or the initial value, initially) is the first
            parameter, followed by all non-sequences."""
            if axis == -1 or axis == 1:
                return T.dot(Wt[0], D).reshape((1, wrows, wcols - 1))
            elif axis == 0:
                return T.dot(Wt[0].T, D).T.reshape((1, wrows - 1, wcols))

        wshape = (wrows, wcols)
        D = construct_d(wshape[axis])
        results, _ = theano.map(fn=fn, sequences=W, non_sequences=D)
        return results

    @staticmethod
    def _diff_operator2D(W, axis):
        wrows, wcols = W.get_value().shape
        if wrows == 784 and wcols == 10:
            # MNIST
            wrows = wcols = 28
            outputs = 10
        elif (wrows % 513) == 0 and wcols == 10:
            # GTZAN
            wcols = wrows / 513
            wrows = 513
            outputs = 10
        else:
            raise NotImplementedError('Diff operator not implemented for this dataset.')

        def construct_d(dim):
            """Costructs the finite difference matrix.

            Params:
                dim: the output dimension, in rows

            Output:
                the finite difference matrix, with shape = (dim, dim-1)
            """
            import scipy.linalg
            import numpy as np
            firstcol = np.zeros(dim, dtype=theano.config.floatX)
            firstcol[0] = -1.
            firstcol[1] = 1.
            firstrow = np.zeros(dim, dtype=theano.config.floatX)
            firstrow[0] = -1.
            return scipy.linalg.toeplitz(firstcol, firstrow)[:, :dim - 1]

        def fn(Wt, D):
            """The function to be passed to theano.map.

            The order of the parameters is fixed by scan: the output of the
            prior call to fn (or the initial value, initially) is the first
            parameter, followed by all non-sequences."""
            Wt = Wt.reshape((wrows, wcols))
            if axis == -1 or axis == 1:
                return T.dot(Wt, D).reshape((wrows * (wcols - 1), ))
            elif axis == 0:
                return T.dot(Wt.T, D).T.reshape(((wrows - 1) * wcols, ))

        wshape = (wrows, wcols)
        D = construct_d(wshape[axis])
        results, _ = theano.map(fn=fn, sequences=W.T, non_sequences=D)
        return results.T

    @staticmethod
    def diff_operator(W, axis):
        ndim = len(W.get_value().shape)
        if ndim == 4:
            return FusedLasso._diff_operator4D(W, axis)
        elif ndim == 2:
            return FusedLasso._diff_operator2D(W, axis)
        else:
            raise NotImplementedError(
                "Diff operator not implemented for ndim={0}".format(ndim))

    def __init__(self, coeffs, axes):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, **kwargs):
        """Returns a theano expression for the cost function.

        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        total_cost : theano.gof.Variable
            coeff * sum(abs(diff(weights)))
            added up for each set of weights.
        """

        assert T.scalar() != 0.  # make sure theano semantics do what I want
        self.get_data_specs(model)[0].validate(data)
        layer_costs = []
        for layer in model.layers:
            layer_name = layer.layer_name
            if layer_name in self.coeffs:
                cost = layer.get_fused_lasso(self.coeffs[layer_name],
                                             self.axes[layer_name])
                if cost != 0.:
                    layer_costs.append(cost)

        if len(layer_costs) == 0:
            rval = T.constant(0., dtype=theano.config.floatX)
            rval.name = '0_fused_lasso'
            return rval
        else:
            total_cost = reduce(operator.add, layer_costs)
        total_cost.name = 'MLP_FusedLasso'

        assert total_cost.ndim == 0

        total_cost.name = 'fused_lasso'

        return total_cost

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False
