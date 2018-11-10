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
    private double epsilon = 0.1, y = 0.8, alpha = 0.02;
    private int actionSpaceSize, netUpdateFrequncy = 1000, miniBatchSize = 128, experienceStoredMaxAmount = 2000000, learningEpochsPerIteration = 10, inputSize = 1;
    Random rnd = new Random();
    public DQN_Learner(MultiLayerConfiguration net){
        this.target = new MultiLayerNetwork(net);
        this.target.init();
        this.pastNetwork = target.clone();
        ExperienceDataSet = new ArrayList<>();
    }
    public void setNetUpdateFrequncy(int netUpdateFrequncy) {
        this.netUpdateFrequncy = netUpdateFrequncy;
    }
    public void setMiniBatchSize(int miniBatchSize) {
        this.miniBatchSize = miniBatchSize;
    }
    public void setExperienceStoredMaxAmount(int experienceStoredMaxAmount) {
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
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }
    public void Learn(int iterations){
        int counter = 0;
        while (counter<iterations){
            if (counter>0&&counter%netUpdateFrequncy==0){updatePastNetwork();}
            environment.reset();
            while (!environment.isEnd()){
                Step s = environment.performAction(produceAction(environment.getCurrentState(), target));
                learnExperienceReplay(s);
                if (ExperienceDataSet.size()==experienceStoredMaxAmount){ExperienceDataSet.remove(0);}
                ExperienceDataSet.add(s);
            }
            counter++;
        }
        environment.reset();
    }
    public void learnExperienceReplay(Step s){
        Step[] ReplayStack = sampleData();
        double[] Q = computeQ(target, s.getBeginStae().getState());
        INDArray input = Nd4j.create(ReplayStack.length+1, inputSize), labels = Nd4j.create(ReplayStack.length+1, actionSpaceSize);
        if (s.isTerminate()){Q[s.getA()] = s.getR();}
        else{ Q[s.getA()] = (1-alpha)*Q[s.getA()]+alpha*(s.getR()+y*produceActionGreedy(s.getEndState(), pastNetwork)); }
        input.put(0, s.getBeginStae().getState());
        for (int i = 0; i <Q.length ; i++) { labels.putScalar(new int[]{0, i}, Q[i]); }
        for (int i = 0; i < ReplayStack.length; i++) {
            input.put(i+1, ReplayStack[i].getBeginStae().getState());
            Q = computeQ(target, ReplayStack[i].getBeginStae().getState());
            if (ReplayStack[i].isTerminate()){Q[ReplayStack[i].getA()] = ReplayStack[i].getR();}
            else{ Q[ReplayStack[i].getA()] = (1-alpha)*Q[ReplayStack[i].getA()]+alpha*(ReplayStack[i].getR()+y*produceActionGreedy(ReplayStack[i].getEndState(), pastNetwork)); }
            for (int j = 0; j < Q.length; j++) {
                labels.putScalar(new int[]{i+1, j}, Q[j]);
            }
        }
        DataSet data = new DataSet(input, labels);
        for (int i = 0; i < learningEpochsPerIteration; i++) {
            target.fit(data);
        }
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
        INDArray out = net.output(input);
        double[] Q = new double[actionSpaceSize];
        for (int i = 0; i < actionSpaceSize; i++) { Q[i] = out.getDouble(i); }
        return Q;
    }
    public int produceAction(State s, MultiLayerNetwork net){
        if (rnd.nextDouble()<epsilon){return rnd.nextInt(actionSpaceSize);}
        return getMaxId(computeQ(net, s.getState()));

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
    public MultiLayerNetwork getTargetNetwork(){return target;}
    private void updatePastNetwork(){
        this.pastNetwork = this.target.clone();
    }
}