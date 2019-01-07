package QLearning_examples.PointRouter;
import ActorCriticLearning.A2C_Learner;
import DQN_learning.Learner.Learner;
import Tools.Linear;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.Random;

public class Router {
    public static Random rnd = new Random();
    public static int n = 3;

    public static void main(String[] args) throws Exception {
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true).setMaximumDeviceCache(2L * 1024L * 1024L * 1024L).allowCrossDeviceAccess(true);
//        for (int i = 0; i < 20; i++) {
            Plane plane = new Plane(n,  4 * n * n);
//            DQN_TD_Learner learner = new DQN_TD_Learner(config(4), QLearner.InputType.Dense, 4);
//            A2C_Learner learner = new A2C_Learner(configPolicy(4), configValue(), Learner.InputType.Dense, Learner.InputType.Dense);
            A2C_Learner learner = new A2C_Learner(configPolicy(4), Learner.InputType.Dense, 8);
            learner.setEnvironment(plane);
            learner.setActionSpaceSize(4);
            learner.setY(0.95);
            learner.setAlpha(0.05);
            learner.setScoreListener(1);
            learner.setRewardScaler(1.0);
            learner.setLearningEpochsPerIterationValue(32);
            learner.Learn(200000);
            learner.getPolicy().save(new File("PointRouter_Wall_policy_" + (n) + "x" + (n) + ".net"));
            learner.getValue().save(new File("PointRouter_Wall_value_" + (n) + "x" + (n) + ".net"));
            n++;
//        }
    }

    static MultiLayerConfiguration configPolicy(int actions) {
        int i = 0;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .l2(0.0005)
                .seed((new Random().nextInt()))
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(1.0))
                .biasUpdater(new Sgd(1.0))
                .biasInit(0.0)
                .list()
                .layer(i, new DenseLayer.Builder().activation(Activation.TANH).nIn(4)
                        .nOut(10).build())
                .layer(i+1, new DenseLayer.Builder().nIn(10).nOut(20).activation(Activation.TANH).build())
                .layer(i+2, new DenseLayer.Builder().activation(Activation.SOFTMAX).nIn(20)
                        .nOut(actions).build())
//                .setInputType(InputType.convolutional(n, n, 1))
                .backprop(true).pretrain(false).build();
        return conf;
    }
    static MultiLayerConfiguration configValue() {
        int i = -2;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .l2(0.0005)
                .seed((new Random().nextInt()))
                .weightInit(WeightInit.XAVIER)
                .updater(new RmsProp(0.005))
                .biasUpdater(new RmsProp(0.005))
                .biasInit((Math.random() - 0.5) / 16.0)
                .list()
//                .layer(i, new ConvolutionLayer.Builder(3, 3).nIn(1).nOut(4).build())
                .layer(i+2, new DenseLayer.Builder().activation(Activation.TANH).nIn(4)
                        .nOut(25).build())
                .layer(i+3, new OutputLayer.Builder()
                        .nOut(1)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .build())
//                .setInputType(InputType.convolutional(n, n, 1))
                .backprop(true).pretrain(false).build();
        return conf;
    }
}