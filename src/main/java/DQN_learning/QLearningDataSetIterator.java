package DQN_learning;

import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;

public class QLearningDataSetIterator extends BaseDatasetIterator {
    public QLearningDataSetIterator(int batch, int numExamples, DataSetFetcher fetcher) {
        super(batch, numExamples, fetcher);
    }
}
