package QLearning_examples.Maze;
import DQN_learning.Learner.Learner;
import DQN_learning.MonteCarlo.DQN_MC_Learner;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
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
    public static int h = 3, w = 3, n = 6;
    public static void main(String[] args) throws Exception {
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true).setMaximumDeviceCache(2L * 1024L * 1024L * 1024L).allowCrossDeviceAccess(true);
        h = n; w = n;
        Maze maze = new Maze(2*h+1, 2*w+1, 2*n*n);
        System.out.println(maze.toString());
        DQN_MC_Learner learner = new DQN_MC_Learner(config(4),  Learner.InputType.Convolution, 4);
        learner.setEnvironment(maze);
        learner.setActionSpaceSize(4);
        learner.setMinEpsilon(0.16);
        learner.setY(0.8);
        learner.setEpsilonUpdateTime(128);
        learner.setEpsilonDecay(0.9);
        learner.setScoreListener(5);
        learner.setRewardScaler(1.0);
        learner.setEpochsForEvaluation(8);
        learner.setLearningEpochsPerIteration(10);
        learner.Learn(2000, 1);
        learner.getTargetNetwork().save(new File("Maze_"+(2*h+1) + "x"+(2*w+1)+".net"));
    }
    static MultiLayerConfiguration config(int actions){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .l2(0.0005)
                .seed((new Random().nextInt()))
                .weightInit(WeightInit.XAVIER)
                .updater(new RmsProp(0.005))
                .biasUpdater(new RmsProp(0.005))
                .biasInit((Math.random()-0.5)/16.0)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5).nIn(1).nOut(4).build())
                .layer(1, new ConvolutionLayer.Builder(4, 4).nIn(4).nOut(1).build())
//                .layer(3, new SubsamplingLayer.Builder().kernelSize(2, 2).stride(1, 1).build())
                .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(40).build())
                .layer(3, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(20).build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
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
