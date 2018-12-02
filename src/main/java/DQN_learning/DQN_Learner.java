package DQN_learning;
import Tools.Pair;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DQN_Learner {
    public enum InputType{
        Covolution, Dense;
        public int GetId(InputType type){
            if (type.equals(Covolution))return 0;
            else if (type.equals(Dense))return 1;
            else return -1;
        }
    }
    public enum NetworkType{
        OneNetwork, MultiNetwork
    }
    private double rewardScaler = 1.0;
    public InputType networkType;
    private boolean useGPU = false;
    //GPU learning doesn`t work yet;
    private Network target, pastNetwork;
    private List<Step> ExperienceDataSet;
    private Environment environment;
    private double epsilon = 1.0, minEpsilon = 0.1, epsilonDecay = 0.9;
    private int batchLearningSize = 8;
    private double y = 0.95;
    private int actionSpaceSize, epsilonUpdateTime = 200, netUpdateFrequncy = 1000, miniBatchSize = 128, experienceStoredMaxAmount = 200000, learningEpochsPerIteration = 10, inputSize = 1, scoreListener = 100;
    Random rnd = new Random();
    public DQN_Learner(MultiLayerConfiguration net, InputType netType, NetworkType networkType, int actionSpaceSize){
        this.actionSpaceSize = actionSpaceSize;
        this.target = new Network(networkType , net, actionSpaceSize, useGPU);
        this.networkType = netType;
        this.pastNetwork = target.clone();
        ExperienceDataSet = new ArrayList<>();
        try { (new File("Log.txt")).delete(); Files.createFile(Paths.get("Log.txt")); Files.write(Paths.get("Log.txt"), "".getBytes(), StandardOpenOption.WRITE); } catch (IOException e) { }
    }
    public void setRewardScaler(double rewardScaler) {
        this.rewardScaler = rewardScaler;
    }
    public double getRewardScaler() {
        return rewardScaler;
    }
    public int getBatchLearningSize() {
        return batchLearningSize;
    }
    public double getY() {
        return y;
    }
    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }
    public void setBatchLearningSize(int batchLearningSize) {
        this.batchLearningSize = batchLearningSize;
    }
    public void setEpsilonDecay(double epsilonDecay) {
        this.epsilonDecay = epsilonDecay;
    }
    public void setUseGPU(boolean useGPU) {
        this.useGPU = useGPU;
    }
    public void setScoreListener(int scoreListener){this.scoreListener=scoreListener;}
    public void setInputSize(int size){this.inputSize = size;}
    public void setEpsilonUpdateTime(int epsilonUpdateTime) {
        this.epsilonUpdateTime = epsilonUpdateTime;
    }
    public void setNetUpdateFrequncy(int netUpdateFrequncy) {
        this.netUpdateFrequncy = netUpdateFrequncy;
    }
    public void setMiniBatchSize(int miniBatchSize) {
        this.miniBatchSize = miniBatchSize;
    }
    public void setExperienceStoredMaxAmount(int experienceStoredMaxAmount) throws Exception {
        if (experienceStoredMaxAmount<4){throw new Exception("Unable to set values smaller than 4 for Experience replay stack size");}
        this.experienceStoredMaxAmount = experienceStoredMaxAmount;
        this.ExperienceDataSet = new ArrayList<>(experienceStoredMaxAmount);
    }
    public void setLearningEpochsPerIteration(int learningEpochsPerIteration) {
        this.learningEpochsPerIteration = learningEpochsPerIteration;
    }
    public void setEnvironment(Environment env){
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
    public void Log(String txt){
        txt+="\n";
        try {
            Files.write(Paths.get("Log.txt"), txt.getBytes(), StandardOpenOption.APPEND);
        }catch (Exception e){}
    }
    public void learnExperienceReplay(Step s){
        Step[] ReplayStack = sampleData();
        ReplayStack[ReplayStack.length-1] = s;
        target.LearnExperienceReplay(ReplayStack, this);
    }
    public double getMaxQ(double[] Q){
        double u  = Q[0];
        for (int i = 0; i < Q.length; i++) {
            if (u < Q[i])u=Q[i];
        }
        return u;
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
    public int produceAction(State s, Network net){
        if (rnd.nextDouble()<epsilon){return rnd.nextInt(actionSpaceSize);}
        return produceActionGreedy(s, net);
    }
    public int produceActionGreedy(State s, Network net){
        if (networkType.equals(InputType.Covolution))
            return getMaxId(net.computeQ(Nd4j.create(s.getConvVersion())));
        else if (networkType.equals(InputType.Dense))
            return getMaxId(net.computeQ(Nd4j.create(s.getState())));
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
    public int getMiniBatchSize() {
        return miniBatchSize;
    }
    public int getLearningEpochsPerIteration() {
        return learningEpochsPerIteration;
    }
    public Network getTargetNetwork(){return target;}
    public Network getPastNetwork(){return pastNetwork;}
    private void updatePastNetwork(){
        this.pastNetwork = this.target.clone();
    }
}