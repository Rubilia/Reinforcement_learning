package DQN_learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class QLearningDataFetcher extends BaseDataFetcher {
    private List<double[][][]> Inputs;
    private List<double[]> InputsDense;
    private List<double[]> Outputs;
    DQN_Learner.InputType type;
    public QLearningDataFetcher(int n, DQN_Learner.InputType type){
        if (type.equals(DQN_Learner.InputType.Covolution)) Inputs = new ArrayList<>(n);
        else if (type.equals(DQN_Learner.InputType.Dense))InputsDense  = new ArrayList<>(n);
        Outputs = new ArrayList<>(n);
        this.type = type;
    }
    public void addConv(double[][][] input, double[] output){
        Inputs.add(input);
        Outputs.add(output);
    }
    public void AddNewValueDense(double[] input, double[] output){
        ArrayList<double[]> inp = new ArrayList<>(Inputs.size()+1);
        inp.addAll(InputsDense);
        inp.add(input.clone());
        ArrayList<double[]> out = new ArrayList<>(Outputs.size()+1);
        out.addAll(Outputs);
        out.add(output);
        InputsDense = inp;
        Outputs = out;
    }
    public void addNewValueConv(double[][][] input, double[] output){
        ArrayList<double[][][]> inp = new ArrayList<>(Inputs.size()+1);
        inp.addAll(Inputs);
        inp.add(input.clone());
        ArrayList<double[]> out = new ArrayList<>(Outputs.size()+1);
        out.addAll(Outputs);
        out.add(output);
        Inputs = inp;
        Outputs = out;
    }
    public DataSet getDataSet(){
        if (type.equals(DQN_Learner.InputType.Dense)){
            double[][] inputs = new double[InputsDense.size()][0], outputs = new double[Outputs.size()][0];
            for (int i = 0; i < InputsDense.size(); i++) {
                inputs[i] = InputsDense.get(i);
                outputs[i] = Outputs.get(i);
            }
            return new DataSet(Nd4j.create(inputs), Nd4j.create(outputs));
        }
        else if (type.equals(DQN_Learner.InputType.Covolution)){
            double[][] outputs = new double[Outputs.size()][0];
            double[][][][] inputs = new double[Inputs.size()][0][0][0];
            for (int i = 0; i < Inputs.size(); i++) {
                inputs[i] = Inputs.get(i);
                outputs[i] = Outputs.get(i);
            }
            return new DataSet(Nd4j.create(inputs), Nd4j.create(outputs));
        }
        return new DataSet();
    }
    @Override
    public void fetch(int numExamples) {
        double[][][][] featureData = new double[numExamples][0][0][0];
        double[][] labelData = new double[numExamples][0], inputDense = new double[numExamples][0];
        int actualExamples = 0;
        for (int i = 0; i < numExamples; i++, cursor++) {
            if (type.equals(DQN_Learner.InputType.Covolution)){
                if (!(this.cursor<Inputs.size())) break;
                featureData[i] = Inputs.get(cursor);
            }
            else if (type.equals(DQN_Learner.InputType.Dense)){
                if (!(this.cursor<InputsDense.size())) break;
                inputDense[i] = InputsDense.get(cursor);
            }
            labelData[i] = Outputs.get(cursor);
            actualExamples++;
        }
        if (actualExamples < numExamples) {
            featureData = Arrays.copyOfRange(featureData, 0, actualExamples);
            if (type.equals(DQN_Learner.InputType.Covolution)) labelData = Arrays.copyOfRange(labelData, 0, actualExamples);
            else if (type.equals(DQN_Learner.InputType.Dense)) inputDense = Arrays.copyOfRange(inputDense, 0, actualExamples);
        }
        INDArray features = (type.equals(DQN_Learner.InputType.Covolution))?Nd4j.create(featureData):Nd4j.create(inputDense);
        INDArray labels = Nd4j.create(labelData);
        curr = new DataSet(features, labels);
    }
    @Override
    public void reset(){
        super.reset();
    }
}
