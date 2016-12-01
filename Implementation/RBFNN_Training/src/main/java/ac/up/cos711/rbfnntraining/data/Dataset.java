package ac.up.cos711.rbfnntraining.data;

import ac.up.cos711.rbfnntraining.data.util.IncorrectFileFormatException;
import ac.up.cos711.rbfnntraining.data.util.TrainingTestingTuple;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Class representing a dataset for neural network training. This class is also
 * used by functions deriving from {@link NetworkError} for the purpose of
 * providing sample data to quantify network classification error.
 * <br><br>
 * Files are required to be in the following format:
 * <br>
 * <pre>
 * i=input_count
 * h=hidden_count
 * t=target_count
 * [pattern1]
 * ...
 * [patternN]</pre>
 * <br>
 * where every value in [patternK] is separated by a white space.
 * <br>
 * The dataset will will assume that NN architecture will only consist of 3
 * layers. An input, hidden and output layer.
 *
 * @author Abrie van Aardt
 */
public class Dataset implements Iterable {

    /**
     * A dataset object can only be instantiated via a call to the static
     * {@link Dataset#fromFile(java.lang.String)} method, or a call to
     * {@link Dataset#split(double)} on an existing Dataset object.
     */
    private Dataset() {
    }

    //TODO: data is scaled to [-1,1]. Double check that [root(3), root(3)] is
    //not preferred
    public static Dataset fromFile(String resourceName)
            throws FileNotFoundException, IncorrectFileFormatException {

        Scanner scanner;
        Dataset dataset = new Dataset();
        ClassLoader classLoader = Dataset.class.getClassLoader();
        scanner = new Scanner(classLoader.getResourceAsStream(resourceName));
        scanner.useDelimiter("\\n|\\r\\n|\\r|\\s");

        //try to parse the number of inputs and targets from the file
        try {
            if (scanner.findInLine("i") == null) {
                throw new IncorrectFileFormatException();
            }

            dataset.inputCount = Integer.parseInt(scanner.findInLine("\\d+"));
            scanner.nextLine();

            if (scanner.findInLine("h") == null) {
                throw new IncorrectFileFormatException();
            }

            dataset.hiddenCount = Integer.parseInt(scanner.findInLine("\\d+"));
            scanner.nextLine();

            if (scanner.findInLine("t") == null) {
                throw new IncorrectFileFormatException();
            }

            dataset.targetCount = Integer.parseInt(scanner.findInLine("\\d+"));
            scanner.nextLine();

        }
        catch (NumberFormatException e) {
            throw new IncorrectFileFormatException();
        }

        //demarshal the data patterns
        double[] inputs = new double[dataset.inputCount];
        double[] targets = new double[dataset.targetCount];

        while (scanner.hasNextDouble()) {
            for (int i = 0; i < inputs.length; i++) {
                inputs[i] = scanner.nextDouble();
            }

            for (int i = 0; i < targets.length; i++) {
                targets[i] = scanner.nextDouble();
            }

            Pattern p = new Pattern();
            p.setInputs(inputs);
            p.setTargets(targets);

            dataset.data.add(p);
        }

        Logger logger = Logger.getLogger(Dataset.class.getName());
        logger.log(Level.INFO, "Loaded {3} pattern(s) with {1} input(s) "
                + "and {2} class(es) from dataset: {0}.", new Object[]{
                    resourceName.substring(resourceName.lastIndexOf('/') + 1),
                    dataset.inputCount,
                    dataset.targetCount,
                    dataset.size()
                });

        logger.log(Level.INFO, "{0} has an optimal 3-layered NN configuration"
                + " requiring {1} hidden units.", new Object[]{
                    resourceName.substring(resourceName.lastIndexOf('/') + 1),
                    dataset.hiddenCount
                });

        dataset.shuffle();

        return dataset;
    }

    @Override
    public Iterator<Pattern> iterator() {
        return data.iterator();
    }

    /**
     * Partitions the dataset. This method provides
     * respective views for the training and testing datasets. Views are
     * backed by the same underlying dataset.
     *
     * @param ratios the ratio in which the dataset should be divided
     * @return List of Datasets
     */
    public List<Dataset> split(double... ratios) {

        if (Arrays.stream(ratios).sum() != 1.0)
            throw new IllegalArgumentException("Ratios in call to split dataset do not add up to 1");

        List<Dataset> datasets = new ArrayList<>();

        int lowerIndex = 0;
        int upperIndex;

        for (int i = 0; i < ratios.length; i++) {
            Dataset dataset = new Dataset();

            dataset.inputCount = inputCount;
            dataset.hiddenCount = hiddenCount;
            dataset.targetCount = targetCount;

            upperIndex = lowerIndex + (int) (ratios[i] * data.size());
            dataset.data = data.subList(lowerIndex, upperIndex);
            lowerIndex = upperIndex;

            datasets.add(dataset);
        }

        String parts = Arrays.toString(Arrays.stream(ratios).map(r -> r * 100.0).toArray());
        
        LOGGER.log(Level.INFO, "Split dataset into {0} part(s): {1}",
                new Object[]{
                    ratios.length,
                    parts
                });
        
        return datasets;
    }

    public Dataset shuffle() {
        Collections.shuffle(data, random);
        return this;
    }

    public int size() {
        return data.size();
    }

    public int getInputCount() {
        return inputCount;
    }

    public int getTargetCount() {
        return targetCount;
    }

    public int getHiddenCount() {
        return hiddenCount;
    }
    
    public String getDatasetName(){
        return name;        
    }   
    
    public void setDatasetName(String _name){
        name = _name;
    }    

    private String name;
    private List<Pattern> data = new ArrayList<>();
    private int inputCount;
    private int hiddenCount;
    private int targetCount;
    private final Random random = new Random(System.nanoTime());
    private static final Logger LOGGER = Logger.getLogger(Dataset.class.getName());
}
