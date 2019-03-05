package NeuralNetwork;

import java.io.IOException;

import ReadData.DataSet;
import ReadData.MNISTdataset;
import ReadData.TestCase;

public class DigitRecognition {
	public static void main(String [] args) throws IOException
	{
		int nlayers = 1;
		int [] sizes = new int[nlayers+2];
		int R = 10, C = 10;
		sizes[0] = R*C; // inputs - 1 due to bias unit
		sizes[1] = 28;
		sizes[2] = 10;

//		MNISTdataset dataSet = new MNISTdataset("train-images.idx3-ubyte",
//										"train-labels.idx1-ubyte");
//		dataSet.preprocess(R,C);
//		Nnet neuralNetwork = new Nnet(nlayers,sizes);
//		neuralNetwork.train(dataSet, 10000, 0.3, 0.1);

		Nnet neuralNetwork = new Nnet("MNistNN10-4.txt");
		
		MNISTdataset dataSetTest = new MNISTdataset("t10k-images.idx3-ubyte",
				"t10k-labels.idx1-ubyte");

		dataSetTest.preprocess(R,C);
		TestCase [] t = dataSetTest.getTrainingCases(); // dataSetTest
		int tot =  t.length, good = 0;
		for (int i = 0; i < t.length; i++)
		{
			TestCase a = t[i];
			int eval = neuralNetwork.evaluate(a);
			System.out.println("Expected "+ a.getTestLabel()+" Found "+eval);
			if (eval==a.getTestLabel())
				good += 1;
		}
		System.out.println("Accuracy " + (double)good  / tot);
//		neuralNetwork.save("MNistNN10-4.txt");
	}
}
