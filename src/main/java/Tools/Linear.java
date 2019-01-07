package Tools;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;

public class Linear implements ILossFunction {
    private INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr;
        // This is the output of the neural network, the y_hat in the notation above
        //To obtain y_hat: pre-output is transformed by the activation function to give the output of the neural network
        INDArray output = activationFn.getActivation(preOutput.dup(), true);
        //The score is calculated as the sum of (y-y_hat)^2 + |y - y_hat|
        scoreArr = output;
        if (mask != null) {
            scoreArr.muliColumnVector(mask);
        }
        return scoreArr;
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
        INDArray dldy = labels.sub(labels).add(1.0);
        INDArray dLdPreOut = activationFn.backprop(preOutput.dup(), dldy).getFirst();
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
        return "Linear";
    }

    @Override
    public String toString() {
        return "Linear()";
    }

    public boolean equals(Object o) {
        if (o == this) return true;
        if (!(o instanceof Linear)) return false;
        final Linear other = (Linear) o;
        if (!other.canEqual(this)) return false;
        return true;
    }

    protected boolean canEqual(Object other) {
        return other instanceof Linear;
    }
}
