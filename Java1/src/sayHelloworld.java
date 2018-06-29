import java1.package1.helloworld;
import java1.package1.helloworld.*;

public class sayHelloworld {

	public static void main(String[] args) {
		helloworld h = new helloworld();// 第一层类
		father f = h.new father();// 第二层类
		f.print();
	}

}
