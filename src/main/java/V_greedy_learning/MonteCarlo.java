package V_greedy_learning;

import Tools.Pair;

import java.util.*;

import static Tools.Graph.buildGraphRW_Vs;

/*
There`s code for run Random walk, evaluate value function V(s) using MonteCarlo algorithm and greedy policy improvements.
zeroReward - reward agent earns when enters left final state.
endReward - reward agent earns when enters right final state.
alpha - in this case equals 1/k, where k is the iterations number in the epoch. also could be a constant or depend on time(in this case a should meet following conditions:
1. alpha(1)+alpha(2)+alpha(3)+...=infinity
2.alpha(1)^2+alpha(2)^2+alpha(3)^2+... < infinity
(sums from 1 to infinity))
y - discount factor using in policy. Usually between 0 and 1
i - number of current epoch
 */
public class MonteCarlo {
    static List<Double> values = new ArrayList<>(), realValues;
    static double zeroRevard = -1.0, endReward = 1.0;
    static long iterationsAmount = 1000, epochsAmount = 2;
    static int iterations = 0, statesAmount = 50;
    static double alpha = 1.0, y = 0.9;
    static int i = 0;
    public static void main(String[] args) {
        System.out.println("Below written values for V(s) for Random Walk. Each learning iteration is highlighted by #n===, n - iteration id");
        values = create(statesAmount);
        realValues = create(statesAmount);
        System.out.println("#1=============================================");
        for (i = 0; i < epochsAmount; i++) {
            evaluate.run();
            realValues.clear();
            realValues.addAll(values);
            values = create(statesAmount);
            iterations=0;
            alpha=1.0;
            if (i==0||i==epochsAmount-1){
                buildGraphRW_Vs(realValues, i+1);
            }
            if (i < epochsAmount-1) System.out.println("#"+(i+2)+"=============================================");
        }
    }
    static List<Double> create(int n){
        List<Double> ret = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            ret.add(0.0);
        }
        return ret;
    }

    static Runnable evaluate = new Runnable() {
        //Random walk
        Random rnd = new Random();
        Map<Integer, Pair<Double, Integer>> map = new HashMap<>();
        int currentPos;
        void init(){
            map = new HashMap<>();
            for (int i = 0; i < statesAmount; i++) { map.put(i, new Pair<>(0.0, 0)); }
            currentPos = rnd.nextInt(statesAmount);
        }
        @Override
        public void run() {
            init();
            while (iterations<iterationsAmount){
                while (currentPos!=-1&&currentPos!=statesAmount){
                    update(move());
                }
                applyResults(map);
                init();
            }
        }

        Pair<Double, Integer> move(){
            int ret = currentPos;
            boolean left;
            if (currentPos==0||currentPos==statesAmount-1) {
                left = currentPos==0?true:false;
            }
            else{
                if (realValues.get(currentPos-1)-realValues.get(currentPos+1)==0.0) {
                    left=rnd.nextBoolean();
                }
                else {
                    left = realValues.get(currentPos-1)>realValues.get(currentPos+1);
                }
            }
            if (left){
                currentPos--;
                if (currentPos==-1){return new Pair<>(zeroRevard, ret);}
                else {return new Pair<>(0.0, ret);}
            }
            else{
                currentPos++;
                if (currentPos==statesAmount){return new Pair<>(endReward, ret);}
                else {return new Pair<>(0.0, ret);}
            }
        }

        public void update(Pair<Double, Integer> reward){
            for (int i = 0; i < statesAmount; i++) {
                Pair<Double, Integer> data = map.get(i);
                if (data.getValue()>0&&reward.getValue()==i)
                {
                    data.setValue(0);
                }
                if (data.getValue()==0&&i!=reward.getValue()){continue;}
                Pair<Double, Integer> ret = new Pair<>(data.getKey() + Math.pow(y, data.getValue() + 1) * reward.getKey(), data.getValue() + 1);
                map.put(i, ret);
            }
        }
    };
    synchronized static void applyResults(Map<Integer, Pair<Double, Integer>> data){
        iterations++;
        for (int index: data.keySet()){
            values.set(index, values.get(index)+1/alpha*(data.get(index).getKey()-values.get(index)));
            alpha++;
        }
    }
}
