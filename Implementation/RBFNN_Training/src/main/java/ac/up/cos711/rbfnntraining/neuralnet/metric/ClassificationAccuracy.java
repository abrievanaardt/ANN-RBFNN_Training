package ac.up.cos711.rbfnntraining.neuralnet.metric;

import ac.up.cos711.rbfnntraining.data.Dataset;
import ac.up.cos711.rbfnntraining.data.Pattern;
import ac.up.cos711.rbfnntraining.util.UnequalArgsDimensionException;
import ac.up.cos711.rbfnntraining.neuralnet.util.ThresholdOutOfBoundsException;
import java.util.Iterator;
import ac.up.cos711.rbfnntraining.neuralnet.IRBFNeuralNet;

/**
 * This class measures the % of correctly classified data patterns. The decision
 * of whether a particular pattern belongs to any class is influenced by the
 * RIGOR parameter. RIGOR should always be in the range [0, (Max_t - Min_t)/2]
 * => [0,0.4] in the case of 0.1 and 0.9 targets.
 *
 * @author Abrie van Aardt
 */
public class ClassificationAccuracy implements INetworkError {

    public ClassificationAccuracy() {

    }

    public ClassificationAccuracy(double _rigor) throws ThresholdOutOfBoundsException {
        if (_rigor < 0 || _rigor > 0.4)
            throw new ThresholdOutOfBoundsException();
        RIGOR = _rigor;
    }

    /**
     * Calculates the % of patterns that were correctly classified by the
     * network on the particular dataset.
     *
     * @param network
     * @param testingSet
     * @return % of patterns correctly classified
     * @throws UnequalInputWeightException
     * @throws UnequalArgsDimensionException
     */
    @Override
    public double measure(IRBFNeuralNet network, Dataset testingSet)
            throws UnequalArgsDimensionException {

        int correctClassCount = 0;

        Iterator<Pattern> testIter = testingSet.iterator();
        while (testIter.hasNext()) {
            Pattern p = testIter.next();
            double[] outputs = network.classify(p.getInputs());
            double[] targets = p.getTargets();
            int correctNodeCount = 0;

            for (int i = 0; i < outputs.length; i++) {
                if (isCorrectClass(targets[i], outputs[i]))
                    ++correctNodeCount;
            }

            if (correctNodeCount == outputs.length)
                ++correctClassCount;
        }

        double percentage = correctClassCount / ((double) testingSet.size()) * 100.0;

        return percentage;
    }

    private static boolean isCorrectClass(double target, double output) {
        return Math.abs(target - output) <= 0.4 - RIGOR;
    }

    private static double RIGOR = 0.2;//must be within bounds
}
