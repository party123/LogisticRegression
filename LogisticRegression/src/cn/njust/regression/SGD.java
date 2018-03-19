package cn.njust.regression;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * 利用随机梯度下降法求解Logistic回归
 *
 */
public class SGD {

	public static void main(String[] args) throws IOException {
		List<String> ex4xList = new ArrayList<String>();
		List<String> ex4yList = new ArrayList<String>();
		ex4xList = readFile("./file/ex4x.dat");
		ex4yList = readFile("./file/ex4y.dat");
		double[][] ex4Array1 = new double[ex4yList.size()][2];
		double[][] ex4Array2 = new double[ex4yList.size()][2];
		for (int i = 0; i < ex4yList.size(); i++) {
			double x1 = Double.valueOf(ex4xList.get(i).split("   ")[0].trim());
			double x2 = Double.valueOf(ex4xList.get(i).split("   ")[1].trim());
			ex4Array1[i][0] = x1;
			ex4Array2[i][0] = x2;
		}
		List<DataNode> ex4List = new ArrayList<DataNode>();
		double xmean1 = getMean(ex4Array1);
		double xstd1 = getStd(ex4Array1);
		double xmean2 = getMean(ex4Array2);
		double xstd2 = getStd(ex4Array2);
		for (int i = 0; i < ex4yList.size(); i++) {
			double x0 = 1.0;
			double x1 = preoperate2(ex4Array1[i][0], xmean1, xstd1);
			double x2 = preoperate2(ex4Array2[i][0], xmean2, xstd2);
			double y = Double.valueOf(ex4yList.get(i));
			DataNode dataNode = new DataNode(x0, x1, x2);// 此处的DataNode类在同包中的Newton类里面定义了
			dataNode.setY(y);
			ex4List.add(dataNode);
		}
		Node thetaNode = regressionSGD(ex4List);
		// 画直线
		System.out.println("（反标准化前）theta0=" + thetaNode.getX0() + " theta1="
				+ thetaNode.getX1() + " theta2=" + thetaNode.getX2());
		reoperate2(thetaNode, xmean1, xstd1, xmean2, xstd2);
	}

	/**
	 * 读取数据文件
	 * 
	 * @param path
	 * @return
	 * @throws IOException
	 */
	public static List<String> readFile(String path) throws IOException {
		File file = new File(path);
		if (!file.exists() || file.isDirectory()) {
			throw new FileNotFoundException();
		}
		BufferedReader br = new BufferedReader(new FileReader(file));
		String temp = null;
		temp = br.readLine();
		List<String> resultList = new ArrayList<String>();
		while (temp != null) {
			resultList.add(temp.trim());
			temp = br.readLine();
		}
		return resultList;
	}

	/**
	 * 梯度下降法 最小二乘
	 * 
	 * @param nodelist
	 * @param xnew
	 * @return
	 */
	public static Node regressionSGD(List<DataNode> nodelist) {
		double theta0 = 0.0;
		double theta1 = 0.0;
		double theta2 = 0.0;
		Node thetaNode = new Node(theta0, theta1, theta2);
		double a = 0.1;
		// double e = 1.0e-10;
		double e = 0.000001;
		int n = nodelist.size();// 训练样本数量
		double j = 0.0;
		double j1 = 0.0;
		int count = 0;
		double jcount = 0.0;
		Random random = new Random();
		do {
			count++;
			j1 = j;
			double jtemp = 0.0;
			for (int k = 0; k < n; k++) {
				double x0 = nodelist.get(k).getX0();
				double x1 = nodelist.get(k).getX1();
				double x2 = nodelist.get(k).getX2();
				DataNode tempNode2 = new DataNode(x0, x1, x2);
				double y = nodelist.get(k).getY();
				double h = Sigmoid(tempNode2, thetaNode);
				jtemp += -(y * Math.log(h) + (1 - y) * Math.log((1 - h)));
			}
			int i = random.nextInt(n);
			System.out.println("随机数为：" + i);
			double x0temp = nodelist.get(i).getX0();
			double x1temp = nodelist.get(i).getX1();
			double x2temp = nodelist.get(i).getX2();
			DataNode tempNode = new DataNode(x0temp, x1temp, x2temp);
			double ytemp = nodelist.get(i).getY();
			double htemp = Sigmoid(tempNode, thetaNode);
			theta0 = theta0 + a * (ytemp - htemp) * x0temp;
			theta1 = theta1 + a * (ytemp - htemp) * x1temp;
			theta2 = theta2 + a * (ytemp - htemp) * x2temp;
			thetaNode.setX0(theta0);
			thetaNode.setX1(theta1);
			thetaNode.setX2(theta2);
			System.out.println("每次求得theta0值为：" + theta0 + " 每次求得theta1值为："
					+ theta1 + " 每次求得theta2值为：" + theta2);
			j = jtemp / n;
			System.out.println("每次求得j值为：" + j + "当前的j1值为" + j1);
			jcount = j - j1;
		} while (Math.abs(jcount) > e);
		System.out.println("迭代的次数为：" + count);
		return thetaNode;
	}

	/**
	 * 求Sigmoid函数h(theta)
	 * 
	 * @param x
	 * @return
	 */
	public static double Sigmoid(DataNode x, Node theta) {

		return 1.0 / (1 + Math.pow(Math.E, -1 * (multiData(x, theta))));
	}

	/**
	 * 求theta*x向量积
	 * 
	 * @param x
	 * @param theta
	 * @return
	 */
	public static double multiData(DataNode x, Node theta) {
		return x.getX0() * theta.getX0() + x.getX1() * theta.getX1()
				+ x.getX2() * theta.getX2();
	}

	/**
	 * 标准化数据，利用x=(x-xmean)/xstd
	 * 
	 * @param x
	 * @param datanode
	 * @return
	 */
	public static double preoperate2(double x, double xmean, double xstd) {
		return (x - xmean) / xstd;

	}

	/**
	 * 反标准化数据，利用newtheta1=theta1/xstd,newtheta0=theta0-(theta1*xmean)/xstd
	 * 
	 * @param x
	 * @param datanode
	 * @return
	 */
	public static Node reoperate2(Node theta, double xmean1, double xstd1,
			double xmean2, double xstd2) {
		double theta0 = theta.getX0();
		double theta1 = theta.getX1();
		double theta2 = theta.getX2();
		double newtheta0 = theta0 - (theta1 * xmean1) / xstd1
				- (theta2 * xmean2) / xstd2;
		double newtheta1 = theta1 / xstd1;
		double newtheta2 = theta2 / xstd2;
		System.out.println("（反标准化后）theta0="+newtheta0 + " theta1=" + newtheta1+" theta2="+newtheta2);
		Node newtheta = new Node(newtheta0, newtheta1, newtheta2);
		return newtheta;

	}

	/**
	 * 求平均数xmean
	 * 
	 * @param datanodes
	 * @return
	 */
	public static double getMean(double[][] datanodes) {
		double sum = 0.0;
		for (int i = 0; i < datanodes.length; i++) {
			double temp = datanodes[i][0];
			sum += temp;
		}
		double xmean = sum / datanodes.length;
		return xmean;
	}

	/**
	 * 求标准差xstd
	 * 
	 * @param datanodes
	 * @return
	 */
	public static double getStd(double[][] datanodes) {
		double xmean = getMean(datanodes);
		double xstd = 0.0;
		double xstd2 = 0.0;
		for (int i = 0; i < datanodes.length; i++) {
			xstd2 += (datanodes[i][0] - xmean) * (datanodes[i][0] - xmean);
		}
		// xstd = Math.sqrt(xstd2 / (datanodes.length - 1));
		xstd = Math.sqrt(xstd2 / datanodes.length);
		return xstd;
	}
}
