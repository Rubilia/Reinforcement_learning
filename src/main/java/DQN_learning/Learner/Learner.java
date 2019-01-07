package DQN_learning.Learner;

import DQN_learning.Environment.Environment;
import DQN_learning.Environment.State;
import DQN_learning.Environment.Step;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Random;

public abstract class Learner {
    private Random rnd = new Random();
    private double rewardScaler = 1.0;
    public static InputType networkType;
    protected MultiLayerNetwork target;
    protected MultiLayerNetwork pastNetwork;
    protected List<Step> ExperienceDataSet;
    protected Environment environment;
    protected double epsilon = 1.0;
    protected double minEpsilon = 0.1;
    protected double epsilonDecay = 0.9;
    protected double y = 0.99;
    protected int actionSpaceSize;
    protected int epsilonUpdateTime = 200;
    protected int experienceStoredMaxAmount = 2000;
    protected int learningEpochsPerIteration = 10;
    protected int scoreListener = 100;

    public enum InputType {
        Convolution, Dense;

        public int GetId(InputType type) {
            if (type.equals(Convolution)) return 0;
            else if (type.equals(Dense)) return 1;
            else return -1;
        }
    }

    public double[] computeQ(State s, MultiLayerNetwork net){
        if (networkType.equals(InputType.Dense))return computeQ(Nd4j.create(s.getState()), net);
        else return computeQ(Nd4j.create(s.getConvVersion()), net);
    }

    public double getMinEpsilon() {
        return minEpsilon;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public double getMaxQ(double[] Q) {
        double u = Q[0];
        for (int i = 0; i < Q.length; i++) {
            if (u < Q[i]) u = Q[i];
        }
        return u;
    }

    public int getActionSpaceSize() {
        return actionSpaceSize;
    }

    public void setRewardScaler(double rewardScaler) {
        this.rewardScaler = rewardScaler;
    }

    public double getRewardScaler() {
        return rewardScaler;
    }

    public abstract double getY();

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void setEpsilonDecay(double epsilonDecay) {
        this.epsilonDecay = epsilonDecay;
    }

    public void setScoreListener(int scoreListener) {
        this.scoreListener = scoreListener;
    }

    public void setEpsilonUpdateTime(int epsilonUpdateTime) {
        this.epsilonUpdateTime = epsilonUpdateTime;
    }

    public void setLearningEpochsPerIteration(int learningEpochsPerIteration) {
        this.learningEpochsPerIteration = learningEpochsPerIteration;
    }

    public void setEnvironment(Environment env) {
        this.environment = env;
    }

    public void setActionSpaceSize(int aSize) {
        this.actionSpaceSize = aSize;
    }

    public void setMinEpsilon(double epsilon) {
        this.minEpsilon = epsilon;
    }

    public void setY(double y) {
        this.y = y;
    }

    public abstract MultiLayerNetwork getTargetNetwork();

    public int getLearningEpochsPerIteration() {
        return learningEpochsPerIteration;
    }

    public int produceAction(State s, MultiLayerNetwork net) {
        if (rnd.nextDouble() < epsilon) {
            return rnd.nextInt(actionSpaceSize);
        }
        return produceActionGreedy(s, net);
    }

    public double[] computeQ(INDArray input, MultiLayerNetwork net) {
        double[] ret = new double[actionSpaceSize];
        INDArray result = net.output(input);
        for (int i = 0; i < actionSpaceSize; i++) {
            ret[i] = result.getDouble(i);
        }
        return ret;
    }

    public abstract int produceActionGreedy(State s, MultiLayerNetwork net);

    public int getMaxId(double[] list) {
        int i = 0;
        for (int j = 1; j < list.length; j++) {
            if (list[i] < list[j]) {
                i = j;
            }
        }
        return i;
    }
}