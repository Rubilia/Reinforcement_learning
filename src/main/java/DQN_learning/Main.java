package DQN_learning;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        int n = 100;
        RandomWalk randomWalk = new RandomWalk(n, 100);
        DQN_Learner learner = new DQN_Learner(config(2));
        learner.setActionSpaceSize(2);
        learner.setEnvironment(randomWalk);
        learner.setExperienceStoredMaxAmount(100);
        learner.setMiniBatchSize(4);
        learner.setNetUpdateFrequncy(20);
        learner.setLearningEpochsPerIteration(1);
        learner.setAlpha(0.02);
        learner.Learn(3);
        buildPolicy(learner.getTargetNetwork(), n);
    }

    static void buildPolicy(MultiLayerNetwork net, int n){
        String[] policy = new String[n];
        policy[0] = getValue(net.output(Nd4j.create(new double[]{-1.0})));
        for (int i = 1; i < n-1; i++) {
            policy[i] = getValue(net.output(Nd4j.create(new double[]{(double)i/(double)n})));
        }
        policy[n-1] = getValue(net.output(Nd4j.create(new double[]{-0.5})));
        String[][] ret = new String[][]{policy};
        Tools.Graph.buildPolicy2D(ret, n, "Policy for RandomWalk using neural nets");
    }

    static String getValue(INDArray Q){
        if (Q.getDouble(0)>Q.getDouble(1)){return "←";}
        else{return "→";}
    }

    static MultiLayerConfiguration config(int actions){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .l2(0.0005)
            .seed((new Random().nextInt()))
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs.Builder().learningRate(.01).build())
            .biasUpdater(new Nesterovs.Builder().learningRate(0.02).build())
            .list()
            .layer(0, new DenseLayer.Builder().nIn(1).activation(Activation.SIGMOID)
                    .nOut(10).build())
            .layer(1, new DenseLayer.Builder().activation(Activation.SIGMOID)
                    .nOut(5).build())

            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                    .nOut(actions)
                    .activation(Activation.TANH)
                    .build())
            .backprop(true).pretrain(false).build();
        return conf;
    }
}