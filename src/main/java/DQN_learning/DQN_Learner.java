package DQN_learning;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DQN_Learner {
    private MultiLayerNetwork target, pastNetwork;
    private List<Step> ExperienceDataSet;
    private Environment environment;
    private double epsilon = 0.2;
    private double y = 0.95;
    private int actionSpaceSize, netUpdateFrequncy = 1000, miniBatchSize = 128, experienceStoredMaxAmount = 200000, learningEpochsPerIteration = 10, inputSize = 1, scoreListener = 100;
    Random rnd = new Random();
    public DQN_Learner(MultiLayerConfiguration net){
        this.target = new MultiLayerNetwork(net);
        this.target.init();
        this.pastNetwork = target.clone();
        ExperienceDataSet = new ArrayList<>();
        target.setEpochCount(2);
        target.setIterationCount(10);
    }
    public void setScoreListener(int scoreListener){this.scoreListener=scoreListener;}
    public void setInputSize(int size){this.inputSize = size;}
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
    public void setActionSpaceSize(int aSize){this.actionSpaceSize = aSize;}
    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }
    public void setY(double y) {
        this.y = y;
    }
    public void Learn(int iterations){
        int counter = 0, localCounter = 0;
        while (counter<iterations){
            environment.reset();
            Step s;
            while (!environment.isEnd()){
                s = environment.performAction(produceAction(environment.getCurrentState(), target));
                learnExperienceReplay(s);
                if (ExperienceDataSet.size()==experienceStoredMaxAmount){ExperienceDataSet.remove(2*rnd.nextInt(ExperienceDataSet.size()/3)+1);}
                ExperienceDataSet.add(s.clone());
                if (localCounter>0&&localCounter%netUpdateFrequncy==0){updatePastNetwork();}
                localCounter++;
            }
            if (counter%scoreListener==0){ System.out.println("epoch #" + counter +": "+environment.getScore(this)); }
            counter++;
        }
        environment.reset();
    }
    public void learnExperienceReplay(Step s){
        Step[] ReplayStack = sampleData();
        double[] Q = computeQ(target, s.getBeginStae().getState());
        if (s.isTerminate()){Q[s.getA()] = s.getR();}
        else{ Q[s.getA()] = s.getR()+y*computeQ(pastNetwork, s.getEndState().getState())[getMaxId(computeQ(target, s.getEndState().getState()))]; }
        INDArray input = Nd4j.create(ReplayStack.length+1, inputSize), labels = Nd4j.create(ReplayStack.length+1, actionSpaceSize);
        INDArray tmp = s.getBeginStae().getState();
        for (int i = 0; i < inputSize; i++) { input.putScalar(new int[]{0, i}, tmp.getDouble(i)); }
        for (int i = 0; i < actionSpaceSize; i++) { labels.putScalar(new int[]{0, 0}, Q[i]); }
        for (int i = 0; i < ReplayStack.length; i++) {
            tmp = ReplayStack[i].getBeginStae().getState();
            for (int j = 0; j < inputSize; j++) { input.putScalar(new int[]{i+1, j}, tmp.getDouble(j)); }
            if (ReplayStack[i].isTerminate()){Q[ReplayStack[i].getA()] = ReplayStack[i].getR();}
            else{ Q[ReplayStack[i].getA()] = ReplayStack[i].getR()+y*computeQ(pastNetwork, ReplayStack[i].getEndState().getState())[getMaxId(computeQ(target, ReplayStack[i].getEndState().getState()))]; }
            for (int j = 0; j < actionSpaceSize; j++) { labels.putScalar(new int[]{i+1, j}, Q[j]); }
        }
        DataSet data = new DataSet(input, labels);
        for (int i = 0; i < learningEpochsPerIteration; i++) {
            target.fit(data);
        }
        double[] Q_new = computeQ(target,  s.getBeginStae().getState());
        double err = getMSE(Q, Q_new);
        err=0;
    }
    private double getMSE(double[] generated, double[] expected){
        double ret = 0.0;
        for (int i = 0; i < generated.length; i++) { ret+=Math.pow(generated[i]-expected[i], 2); }
        return ret;
    }
    private Step[] sampleData(){
        Step[] ret;
        if (ExperienceDataSet.size()<=miniBatchSize){
            ret = new Step[ExperienceDataSet.size()];
            int j = 0;
            for(Step step: ExperienceDataSet){ret[j] = step;j++;}
        }
        else{
            ret = new Step[miniBatchSize];
            List<Integer> indexes = new ArrayList<>(miniBatchSize);
            for (int i = 0; i < miniBatchSize; i++) {
                int index = 0;
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
    public double[] computeQ(MultiLayerNetwork net, INDArray input){
        INDArray output = net.output(input);
        double[] Q = new double[actionSpaceSize];
        for (int i = 0; i < actionSpaceSize; i++) {
            Q[i] = output.getDouble(i);
        }
        return Q;
    }
    public int produceAction(State s, MultiLayerNetwork net){
        if (rnd.nextDouble()<epsilon){return rnd.nextInt(actionSpaceSize);}
        double[] Q = computeQ(net, s.getState());
        return getMaxId(Q);

    }
    public int produceActionGreedy(State s, MultiLayerNetwork net){
        return getMaxId(computeQ(net, s.getState()));
    }
    public int getMaxId(double[] list){
        int i = 0;
        for (int j = 1; j < list.length; j++) {
            if (list[i]<list[j]){i = j;}
        }
        return i;
    }
    public double getMaxQ(MultiLayerNetwork net, INDArray input){
        double[] Q =computeQ(net, input);
        return Q[getMaxId(Q)];
    }
    public MultiLayerNetwork getTargetNetwork(){return target;}
    private void updatePastNetwork(){
        this.pastNetwork = this.target.clone();
    }
}