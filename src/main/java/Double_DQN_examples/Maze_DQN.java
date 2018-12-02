package Double_DQN_examples;

import DQN_learning.DQN_Learner;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Maze_DQN {
    public static Random rnd = new Random();
    public static int h = 3, w = 3, n = 4;
    public static void main(String[] args) throws Exception {
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true).setMaximumDeviceCache(2L * 1024L * 1024L * 1024L).allowCrossDeviceAccess(true);
        h = n; w = n;
        Maze maze = new Maze(2*h+1, 2*w+1, 4*n*n, false);
        System.out.println(maze.toString());
        DQN_Learner learner = new DQN_Learner(config(4), DQN_Learner.InputType.Covolution, DQN_Learner.NetworkType.OneNetwork, 4);
        learner.setEnvironment(maze);
        learner.setActionSpaceSize(4);
        learner.setExperienceStoredMaxAmount(5000);
        learner.setMiniBatchSize(32);
        learner.setUseGPU(false);
        learner.setMinEpsilon(0.08);
        learner.setEpsilonUpdateTime(1024);
        learner.setEpsilonDecay(0.9);
        learner.setScoreListener(1);
        learner.setRewardScaler(5.0);
        learner.setNetUpdateFrequncy(10);
        learner.setLearningEpochsPerIteration(25);
        learner.setInputSize((2*h-1)*(2*w-1));
        learner.Learn(1000);
        learner.getTargetNetwork().getNetwork()[0].save(new File("Maze_"+(2*h+1) + "x"+(2*w+1)+".net"));
    }
    static MultiLayerConfiguration config(int actions){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .l2(0.0005)
                .seed((new Random().nextInt()))
                .weightInit(WeightInit.XAVIER)
                .updater(new RmsProp(0.001))
                .biasUpdater(new RmsProp(0.001))
                .biasInit((Math.random()-0.5)/4.0)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5).nIn(1).nOut(4).build())
//                .layer(1, new ConvolutionLayer.Builder(4, 4).nIn(4).nOut(4).build())
//                .layer(2, new ConvolutionLayer.Builder(3, 3).nIn(4).nOut(2).build())
//                .layer(3, new SubsamplingLayer.Builder().kernelSize(2, 2).stride(1, 1).build())
                .layer(1, new DenseLayer.Builder().activation(Activation.TANH)
                        .nOut(20).build())
                .layer(2, new DenseLayer.Builder().activation(Activation.IDENTITY)
                        .nOut(40).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(actions)
                        .activation(Activation.IDENTITY)
                        .build()).setInputType(InputType.convolutional(2*h-1, 2*w-1, 1))
                .backprop(true).pretrain(false).build();
        return conf;
    }
    public static void GenerateMazeImage(Maze maze) throws IOException {
        BufferedImage image = new BufferedImage(4*n, 4*n, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                int rgb = maze.getMaze()[x][y]==1.0? Color.BLACK.getRGB(): Color.WHITE.getRGB();
                if (x==n-1&&y==n-1){rgb = Color.RED.getRGB();}
                else if (x==0&&y==0){rgb = Color.GREEN.getRGB();}
                image.setRGB(4*x, 4*y, rgb);
                image.setRGB(4*x+1, 4*y, rgb);
                image.setRGB(4*x+2, 4*y, rgb);
                image.setRGB(4*x+3, 4*y, rgb);
                image.setRGB(4*x, 4*y+1, rgb);
                image.setRGB(4*x+1, 4*y+1, rgb);
                image.setRGB(4*x+2, 4*y+1, rgb);
                image.setRGB(4*x+3, 4*y+1, rgb);
                image.setRGB(4*x, 4*y+2, rgb);
                image.setRGB(4*x+1, 4*y+2, rgb);
                image.setRGB(4*x+2, 4*y+2, rgb);
                image.setRGB(4*x+3, 4*y+2, rgb);
                image.setRGB(4*x, 4*y+3, rgb);
                image.setRGB(4*x+1, 4*y+3, rgb);
                image.setRGB(4*x+2, 4*y+3, rgb);
                image.setRGB(4*x+3, 4*y+3, rgb);
            }
        }
        File outputFile = new File("Maze.jpg");
        ImageIO.write(image, "jpg", outputFile);
    }
}
