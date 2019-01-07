package DQN_learning.MonteCarlo;
import DQN_learning.Learner.Learner;
import DQN_learning.Environment.State;
import DQN_learning.Environment.Step;
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

public class DQN_MC_Learner extends Learner {
    //GPU learning doesn`t work yet;
    Random rnd = new Random();
    protected int epochsForEvaluation = 10;
    private int netUpdateFrequncy = 32, miniBatchSize = 16;
    public void setNetUpdateFrequncy(int netUpdateFrequncy) {
        this.netUpdateFrequncy = netUpdateFrequncy;
    }
    public void setMiniBatchSize(int miniBatchSize) {
        this.miniBatchSize = miniBatchSize;
    }
    public void setEpochsForEvaluation(int epochsForEvaluation) {
        this.epochsForEvaluation = epochsForEvaluation;
    }
    public DQN_MC_Learner(MultiLayerConfiguration net, InputType netType, int actionSpaceSize){
        this.actionSpaceSize = actionSpaceSize;
        this.target = new MultiLayerNetwork(net);
        target.init();
        this.networkType = netType;
        this.pastNetwork = target.clone();
        pastNetwork.init();
        ExperienceDataSet = new ArrayList<>();
        try { (new File("Log.txt")).delete(); Files.createFile(Paths.get("Log.txt")); Files.write(Paths.get("Log.txt"), "".getBytes(), StandardOpenOption.WRITE); } catch (IOException e) { }
    }
    public void Learn(int iterations, int limit){
        int counter = 0, localCounter = 0;
        environment.reset();
        Log(environment.toString());
        MC_Return MonteCarlo = new MC_Return(environment.clone(), epochsForEvaluation, target, this);
        while (counter<iterations){
            for (int i = 0; i < limit; i++) {
                if (networkType.equals(InputType.Dense))learnMC_Dense(MonteCarlo.getReturnMulti(epochsForEvaluation));
                else learnMC_Conv(MonteCarlo.getReturnMulti(epochsForEvaluation));
                if (localCounter%epsilonUpdateTime==0&&epsilon*epsilonDecay>=minEpsilon){epsilon*=epsilonDecay;}
                localCounter++;
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
    private void learnMC_Dense(List<Pair<State, double[]>> data){
        double[][] inputs = new double[data.size()][], outputs = new double[data.size()][];
        int i = 0;
        for(Pair<State, double[]> s : data){
            inputs[i] = s.getKey().getState();
            outputs[i] = s.getValue();
            i++;
        }
        DataSet dataSet = new DataSet(Nd4j.create(inputs), Nd4j.create(outputs));
        for (int j = 0; j < learningEpochsPerIteration; j++) {
            target.fit(dataSet);
        }
//        double result = 0;
//        for (int j = 0; j < data.size(); j++) {
//            result = getMSE(outputs[j], computeQ(Nd4j.create(inputs[j]), target));
//        }
//        result++;
    }
    private void learnMC_Conv(List<Pair<State, double[]>> data){
        double[][][][] inputs = new double[data.size()][][][];
        double[][] outputs = new double[data.size()][];
        int i = 0;
        for(Pair<State, double[]> s : data){
            inputs[i] = s.getKey().getConvVersion()[0];
            outputs[i] = s.getValue();
            i++;
        }
        DataSet dataSet = new DataSet(Nd4j.create(inputs), Nd4j.create(outputs));
        for (int j = 0; j < learningEpochsPerIteration; j++) {
            target.fit(dataSet);
        }
        double result = 0;
        for (int j = 0; j < data.size(); j++) {
            result = getMSE(outputs[j], computeQ(Nd4j.create(new double[][][][]{inputs[j]}), target));
        }
        result++;
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
        return this.y;
    }

    @Override
    public MultiLayerNetwork getTargetNetwork() {
        return target;
    }

    @Override
    public int produceActionGreedy(State s, MultiLayerNetwork net) {
        if (networkType.equals(Learner.InputType.Convolution))
            return getMaxId(computeQ(Nd4j.create(s.getConvVersion()), net));
        else if (networkType.equals(Learner.InputType.Dense))
            return getMaxId(computeQ(Nd4j.create(s.getState()), net));
        else
            return rnd.nextInt(actionSpaceSize);
    }

    public static double getMSE(double[] A, double[] B){
        double MSE = 0.0;
        for (int i = 0; i < A.length; i++) {
            MSE+=Math.pow(A[i]-B[i], 2);
        }
        return MSE;
    }
    private Step[] sampleData(){
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
        return ret;
    }

    public int getMaxId(double[] list){
        int i = 0;
        for (int j = 1; j < list.length; j++) {
            if (list[i]<list[j]){i = j;}
        }
        return i;
    }
    public int getMiniBatchSize() {
        return miniBatchSize;
    }
    private void updatePastNetwork(){
        this.pastNetwork = this.target.clone();
    }
}