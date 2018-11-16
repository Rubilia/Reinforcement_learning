package Double_DQN_examples;

import DQN_learning.DQN_Learner;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Maze_DQN {
    public static int h = 3, w = 3, n = 5;
    public static void main(String[] args) throws Exception {
        h = n; w = n;
        Maze maze = new Maze(2*h+1, 2*w+1, 160, true);
        DQN_Learner learner = new DQN_Learner(config(4));
        learner.setEnvironment(maze);
        learner.setActionSpaceSize(4);
        learner.setExperienceStoredMaxAmount(2000);
        learner.setMiniBatchSize(128);
        learner.setEpsilon(0.12);
        learner.setScoreListener(50);
        learner.setNetUpdateFrequncy(1000);
        learner.setLearningEpochsPerIteration(5);
        learner.setInputSize((2*h-1)*(2*w-1)+2);
        learner.Learn(5000);
    }
    static MultiLayerConfiguration config(int actions){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .l2(0.0005)
                .seed((new Random().nextInt()))
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs())
                .biasUpdater(new Nesterovs())
                .biasInit(0.1)
                .list()
                .layer(0, new DenseLayer.Builder().nIn((2*h-1)*(2*w-1)+2).activation(Activation.TANH)
                        .nOut(40).build())
                .layer(1, new DenseLayer.Builder().activation(Activation.TANH)
                        .nOut(20).build())
                .layer(2, new DenseLayer.Builder().activation(Activation.TANH)
                        .nOut(20).build())
                .layer(3, new DenseLayer.Builder().activation(Activation.TANH)
                        .nOut(10).build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
                        .nOut(actions)
                        .activation(Activation.IDENTITY)
                        .build())
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
