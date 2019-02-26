package ReadData;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.Scanner;


/* MNIST dataset, see http://yann.lecun.com/exdb/mnist/ */
public class MNISTdataset extends DataSet {
	int nlabels;
//	static int MAX = 10000; 
	public int readInt(InputStream in) throws IOException
	{
		int res = 0;
		for (int i = 0; i < 4; i++)
			res = (res<<8) + in.read();
		return res;
	}
	/* if the labels are provided, this is assumed a training set */
	public MNISTdataset(String filePathInputs, String filePathLabels) throws IOException 
	{
		InputStream inInputs = new FileInputStream(filePathInputs);
		InputStream inLabels = new FileInputStream(filePathLabels);
		
		
		int magicNumberInputs = readInt(inInputs);
		assert magicNumberInputs == 2051;
		
		int magicNumberLabels = readInt(inLabels);
		assert magicNumberLabels == 2049;
		
		M = readInt(inInputs);
		
		assert M == readInt(inLabels);
		
//		M = Math.min(MAX, M);

		cases = new Image[M];
		int R = readInt(inInputs), C = readInt(inInputs);

		for (int i = 0; i < M; i++)
			cases[i] = new Image(R,C,inInputs,inLabels);
		
		inInputs.close(); inLabels.close();
	}
	
	/* a test set */
	MNISTdataset(String filePathInputs) throws IOException
	{
		InputStream inInputs = new FileInputStream(filePathInputs);
		
		int magicNumberInputs = readInt(inInputs);
		assert magicNumberInputs == 2051;
		
		
		M = readInt(inInputs);
		
		cases = new Image[M];
		
		int R = readInt(inInputs), C = readInt(inInputs);

		for (int i = 0; i < M; i++)
			cases[i] = new Image(R,C,inInputs);
		
		inInputs.close();
	}
	
	public void preprocess(int R2, int C2)
	{		
		for (int i = 0; i < cases.length; i++)
		{
			((Image)cases[i]).reduceImage(R2,C2);
//			((Image)cases[i]).blackWhite();
		}
	}
	
//	public void reduceImages(int R2, int C2)
//	{		
//		for (int i = 0; i < cases.length; i++)
//			((Image)cases[i]).reduceImage(R2,C2);
//	}


}
