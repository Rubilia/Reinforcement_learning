package DQN_learning.TD_Learning;
import DQN_learning.Environment.State;
import DQN_learning.Environment.Step;
import DQN_learning.Learner.Learner;
import Tools.Pair;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DQN_TD_Learner extends Learner {
    //GPU learning doesn`t work yet;
    Random rnd = new Random();
    private int netUpdateFrequncy = 32, miniBatchSize = 16;
    public void setNetUpdateFrequncy(int netUpdateFrequncy) {
        this.netUpdateFrequncy = netUpdateFrequncy;
    }
    public void setMiniBatchSize(int miniBatchSize) {
        this.miniBatchSize = miniBatchSize;
    }

    public DQN_TD_Learner(MultiLayerConfiguration net, InputType netType, int actionSpaceSize){
        this.actionSpaceSize = actionSpaceSize;
        this.target = new MultiLayerNetwork(net);
        this.target.init();
        this.networkType = netType;
        this.pastNetwork = target.clone();
        this.pastNetwork.init();;
        ExperienceDataSet = new ArrayList<>();
        try { (new File("Log.txt")).delete(); Files.createFile(Paths.get("Log.txt")); Files.write(Paths.get("Log.txt"), "".getBytes(), StandardOpenOption.WRITE); } catch (IOException e) { }
    }
    public void Learn(int iterations){
        int counter = 0, localCounter = 0;
        Log(environment.toString());
        while (counter<iterations){
            environment.reset();
            Step s;
            while (!environment.isEnd()){
                s = environment.performAction(produceAction(environment.getCurrentState(), target));
                learnExperienceReplay(s);
                if (ExperienceDataSet.size()==experienceStoredMaxAmount){ExperienceDataSet.remove(2*rnd.nextInt(ExperienceDataSet.size()/3)+1);}
                ExperienceDataSet.add(s.clone());
                localCounter++;
                if (localCounter%netUpdateFrequncy==0){
                    updatePastNetwork();
                }
                if (localCounter%epsilonUpdateTime==0&&epsilon*epsilonDecay>=minEpsilon){epsilon*=epsilonDecay;}
            }
            if (counter%scoreListener==0){
                Pair<String, Boolean> score =(environment).getScore(this);
                Log("epoch #" + counter +": epsilon: " + epsilon + ", " + score.getKey());
                System.out.println("##############################################################");
                System.out.println("epoch #" + counter +": epsilon: " + epsilon + ", " + score.getKey());
                System.out.println("##############################################################");
                if (score.getValue()){
                    return;
                }
            }
            counter++;
        }
        environment.reset();
    }
    public void learnExperienceReplay(Step s){
        Step[] data = sampleData(s);
        double[] Q;
        double[][] Q_Target = new double[data.length][];
        State[] input = new State[data.length];
        for (int i = 0; i < data.length; i++) {
            Q = computeQ(data[i].getBeginState(), target);
            if (data[i].isTerminate()){Q[data[i].getA()] = data[i].getR()*getRewardScaler();}
            else{ Q[data[i].getA()] = data[i].getR()*getRewardScaler()+getY()*getMaxQ(computeQ(data[i].getEndState(), pastNetwork)); }
            input[i] = data[i].getBeginState();
            Q_Target[i] = Q.clone();
        }
        DataSet dataSet;
        if (networkType.equals(InputType.Dense)){
            double[][] inputs = new double[data.length][];
            for (int i = 0; i < data.length; i++) {
                inputs[i] = data[i].getBeginStateDense();
            }
            dataSet = new DataSet(Nd4j.create(inputs), Nd4j.create(Q_Target));
        }
        else{
            double[][][][] inputs = new double[data.length][][][];
            for (int i = 0; i < data.length; i++) {
                inputs[i] = data[i].getBeginStateConv();
            }
            dataSet = new DataSet(Nd4j.create(inputs), Nd4j.create(Q_Target));
        }
        for (int i = 0; i < learningEpochsPerIteration; i++) {
            target.fit(dataSet);
        }
    }
    public void Log(String txt){
        txt+="\n";
        try {
            Files.write(Paths.get("Log.txt"), txt.getBytes(), StandardOpenOption.APPEND);
        }catch (Exception e){}
    }
    public double getMaxQ(double[] Q){
        double u  = Q[0];
        for (int i = 0; i < Q.length; i++) {
            if (u < Q[i])u=Q[i];
        }
        return u;
    }

    @Override
    public double getY() {
        return y;
    }

    @Override
    public MultiLayerNetwork getTargetNetwork() {
        return target;
    }

    public static double getMSE(double[] A, double[] B){
        double MSE = 0.0;
        for (int i = 0; i < A.length; i++) {
            MSE+=Math.pow(A[i]-B[i], 2);
        }
        return MSE;
    }
    private Step[] sampleData(Step s){
        Step[] ret;
        if (ExperienceDataSet.size()<=miniBatchSize){
            ret = new Step[ExperienceDataSet.size()+1];
            int j = 0;
            for(Step step: ExperienceDataSet){ret[j] = step;j++;}
        }
        else{
            ret = new Step[miniBatchSize+1];
            List<Integer> indexes = new ArrayList<>(miniBatchSize);
            for (int i = 0; i < miniBatchSize; i++) {
                int index;
                do {
                    index = rnd.nextInt(ExperienceDataSet.size());
                }while (indexes.contains(index));
                indexes.add(index);
            }
            int j = 0;
            for(int item : indexes){
                ret[j] = ExperienceDataSet.get(item);
                j++;
            }
        }
        ret[ret.length-1] = s.clone();
        return ret;
    }
    public int produceAction(State s, MultiLayerNetwork net){
        if (rnd.nextDouble()<epsilon){return rnd.nextInt(actionSpaceSize);}
        return produceActionGreedy(s, net);
    }
    @Override
    public int produceActionGreedy(State s, MultiLayerNetwork net){
        if (networkType.equals(InputType.Convolution))
            return getMaxId(computeQ(Nd4j.create(s.getConvVersion()), net));
        else if (networkType.equals(InputType.Dense))
            return getMaxId(computeQ(Nd4j.create(s.getState()), net));
        else
            return rnd.nextInt(actionSpaceSize);
    }
    public int getMaxId(double[] list){
        int i = 0;
        for (int j = 1; j < list.length; j++) {
            if (list[i]<list[j]){i = j;}
        }
        return i;
    }
    public MultiLayerNetwork getPastNetwork(){return pastNetwork;}
    public int getMiniBatchSize() {
        return miniBatchSize;
    }
    private void updatePastNetwork(){
        this.pastNetwork = this.target.clone();
    }
}