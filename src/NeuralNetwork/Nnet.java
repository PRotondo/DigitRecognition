package NeuralNetwork;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.Scanner;

import ReadData.DataSet;
import ReadData.TestCase;



public class Nnet {
	/* number of hidden layers */
	int nlayers;
	/* index 0 corresponds to the number of inputs while index nlayers+1 corresponds to the outputs */
	/* the indexes 1 through nlayers correspond to the hidden layers */
	/* size[k] indicates the number of units in the k-th layer, wihtout counting the bias at index 0 */
	int [] sizes;
	/*	The entry Theta[k][i][j] represents the weight from node i (layer k) to node j (layer k+1) */
	/*  The index 0 always corresponds to the bias unit */
	float [][][] Theta;
	/* The evaluated values after FP */
	/* A[k][0] == 1 always as this corresponds to the bias unit */
	float [][] A,Z;
	/* size of the batch to train the Nnetwork */
	int batchSize = 1000;
	// TODO: make class for this
	/* Activation function */
	float activationFunction(float z)
	{
		return (float) (1/(1+Math.exp(-z)));
	}
	float activationFunctionDerivative(float z)
	{
		float gz = this.activationFunction(z);
		return gz*(1.0f-gz);
	}
	/* constructor */
	/* TODO: create arrays */
	Nnet(int nlayers, int [] sizes)
	{
		this.nlayers = nlayers;
		this.sizes = sizes;
		A = new float[nlayers+2][];
		Z = new float[nlayers+2][];
		for (int i = 0; i <= nlayers+1;i++)
		{
			A[i] = new float[sizes[i]+1];
			Z[i] = new float[sizes[i]+1];
		}
	}
	Nnet(int nlayers, int [] sizes, int batchSize)
	{
		this(nlayers,sizes);
		this.batchSize = batchSize;
	}
	void loadInput(float [] input)
	{
		assert (sizes[0]+1 == input.length);
		/* inputs are loaded into the first position */
		/* TODO : the bias is already considered in the DataSet,not so nice*/
		A[0] = input;
	}
	void forwardPropagation()
	{
		for (int k = 0; k <= nlayers; k++)
		{
			/* bias unit */
			A[k+1][0] = 1.0f;
			/* the rest */
			for (int j = 1; j <= sizes[k+1]; j++)
			{
				float z = 0.0f;
				for (int i = 0; i <= sizes[k]; i++)
					z += Theta[k][i][j] * A[k][i];
				A[k+1][j] = activationFunction(z);
				Z[k+1][j] = z;

			}
		}
	}
	/* contribution to the gradient (non normalized) of a test case */
	/* accumulate the results in D */
	void gradientBackPropagationTestCase(float [][][] D, int label)
	{		/* errors */
		float [][] error = new float[nlayers+2][];
		/* fill the arrays */
		for (int k = 0; k <= nlayers+1; k++)
			error[k] = new float[sizes[k]+1];
		/* set errors for the output */
		for (int j = 1; j <= sizes[nlayers+1]; j++)
		{
			error[nlayers+1][j] = A[nlayers+1][j] - (label==j ? 1 : 0);
			for (int i = 0; i <= sizes[nlayers]; i++)
				D[nlayers][i][j] += error[nlayers+1][j] * A[nlayers][i];
		}
		/* go back the net */
		for (int k = nlayers; k >= 1; k--)
		{
			for (int j = 1; j <= sizes[k]; j++)
			{
				error[k][j] = 0.0f;
				for (int i = 1; i <= sizes[k+1]; i++)
					error[k][j] += error[k+1][i] * Theta[k][j][i];
				error[k][j] *= this.activationFunctionDerivative(Z[k][j]);																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																													
				
				for (int i = 0; i <= sizes[k-1]; i++)
					D[k-1][i][j] += error[k][j] * A[k-1][i];
			}
		}
	}

	
	/* compute gradient */
	float [][][] gradient(DataSet dataSet, float lambda)
	{
		
		float [][][] D = new float[nlayers+1][][];
		for (int k = 0;k <= nlayers; k++)
		{
			D[k] = new float[sizes[k]+1][sizes[k+1]+1];
			for (int i = 0; i <= sizes[k]; i++)
				for (int j = 1; j <= sizes[k+1]; j++)
					D[k][i][j] = 0.0f;
		}	
		TestCase [] cases = dataSet.getBatch(batchSize);
		int M = cases.length;
		for (int t = 0; t < M; t++)
		{
			this.loadInput(cases[t].getTestData());
			this.forwardPropagation();
			this.gradientBackPropagationTestCase(D, cases[t].getTestLabel());
//			System.err.println("label = "+cases[t].getTestLabel());
		}
		for (int k = 0;k <= nlayers; k++)
		{
			for (int i = 0; i <= sizes[k]; i++)
				for (int j = 1; j <= sizes[k+1]; j++)
					D[k][i][j] = (D[k][i][j]+(i==0 ? 0.0f:(lambda*Theta[k][i][j])))/M; // 
		}	
		return D;
	}
	/* gradient descent */
	void train(DataSet dataSet, int iter, float alpha, float lambda)
	{
		float epsilon = 0.5f;
		Theta = new float[nlayers+1][][];
		for (int k = 0;k <= nlayers; k++)
		{
			Theta[k] = new float[sizes[k]+1][sizes[k+1]+1];
			for (int i = 0; i <= sizes[k]; i++)
				for (int j = 1; j <= sizes[k+1]; j++)
				{
					Theta[k][i][j] = (float) (Math.random()*epsilon*2 - epsilon);
//					System.err.println(Theta[k][i][j]);
				}
		}
		
//		System.out.println(activationFunction(0));
//		System.out.println(activationFunction(1));
//		System.out.println(activationFunction(-1));

		float [][][] D;
		for (int t = 0; t < iter; t++)
		{
			System.err.println("Iteration #"+t);
//			alpha = Math.min(alpha,1/(float)(t+1));
			D = this.gradient(dataSet, lambda);
			for (int k = 0;k <= nlayers; k++)
				for (int i = 0; i <= sizes[k]; i++)
					for (int j = 1; j <= sizes[k+1]; j++)
					{
						Theta[k][i][j] -= alpha * D[k][i][j];
//						System.err.println(k+" "+i+" "+j+" "+Theta[k][i][j]);
					}
		}
	}
	
	// TODO add function for error

	int evaluate(TestCase testCase)
	{
		this.loadInput(testCase.getTestData());
		this.forwardPropagation();
		int res = 0;
		float certainty = 0.0f;
		for (int i = 1; i <= sizes[this.nlayers+1]; System.out.println(i + " " + A[nlayers+1][i++]))
			if (A[nlayers+1][i]>certainty)
			{
				certainty = A[nlayers+1][i];
				
				res = i;
			}
		
		return res;
	}
	
	public int evaluate(float [] data)
	{
		this.loadInput(data);
		this.forwardPropagation();
		int res = 0;
		float certainty = 0.0f;
		for (int i = 1; i <= sizes[this.nlayers+1]; System.out.println(i + " " + A[nlayers+1][i++]))
			if (A[nlayers+1][i]>certainty)
			{
				certainty = A[nlayers+1][i];
				
				res = i;
			}
		
		return res;
	}
	void save(String filePath) throws IOException
	{
		FileWriter outW = new FileWriter(filePath);
		PrintWriter out = new PrintWriter(new BufferedWriter(outW));
		out.println(this.nlayers);
		for (int i = 0; i < 1+this.nlayers; i++)
			out.print(sizes[i]+" ");
		out.println(sizes[1+nlayers]);
		for (int k = 0;k <= nlayers; k++)
		{
			for (int i = 0; i <= sizes[k]; i++)
				for (int j = 1; j <= sizes[k+1]; j++)
					out.println(Theta[k][i][j]);
		}
		out.close();
	}
	public Nnet(String filePath) throws IOException
	{
		FileReader inR = new FileReader(filePath);
		Scanner in = new Scanner(new BufferedReader(inR));
		nlayers = in.nextInt();
		sizes = new int[nlayers+2];
		for (int i = 0; i <= 1+this.nlayers; i++)
			sizes[i] = in.nextInt();
		Theta = new float[nlayers+1][][];
		for (int k = 0;k <= nlayers; k++)
		{
			Theta[k] = new float[sizes[k]+1][sizes[k+1]+1];
			for (int i = 0; i <= sizes[k]; i++)
				for (int j = 1; j <= sizes[k+1]; j++)
					Theta[k][i][j] = in.nextFloat();
		}
		A = new float[nlayers+2][];
		Z = new float[nlayers+2][];
		for (int i = 0; i <= nlayers+1;i++)
		{
			A[i] = new float[sizes[i]+1];
			Z[i] = new float[sizes[i]+1];
		}
		in.close();
	}
}
