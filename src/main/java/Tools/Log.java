package Tools;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

public class Log implements ILossFunction {
    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        // This is the output of the neural network, the y_hat in the notation above
        //To obtain y_hat: pre-output is transformed by the activation function to give the output of the neural network
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        //The score is calculated as the sum of (y-y_hat)^2 + |y - y_hat|
        INDArray ret = Transforms.log(Transforms.abs(output));
        if (mask != null) {
            ret.muliColumnVector(mask);
        }
        return ret;
    }
    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);

        double score = scoreArr.sumNumber().doubleValue();

        if (average) {
            score /= scoreArr.size(0);
        }

        return score;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = scoreArray(labels, preOutput, activationFn, mask);
        return scoreArr.sum(1);
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        /*
        //NOTE: There are many ways to do this same set of operations in nd4j
        //The following is the most readable for the sake of this example, not necessarily the fastest
        //Refer to the Implementation of LossL1 and LossL2 for more efficient ways
        */
        INDArray dldyhat = Transforms.pow(output, -1); //d(L)/d(yhat) -> this is the line that will change with your loss function

        //Everything below remains the same
        INDArray dLdPreOut = activationFn.backprop(preOutput.dup(), dldyhat).getFirst();
        //multiply with masks, always
        if (mask != null) {
            dLdPreOut.muliColumnVector(mask);
        }

        return dLdPreOut;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(
                computeScore(labels, preOutput, activationFn, mask, average),
                computeGradient(labels, preOutput, activationFn, mask));
    }

    @Override
    public String name() {
        return "Log()";
    }
}
