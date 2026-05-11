

###### scala基本数据类型

| 类型      | 说明                          |
| --------- | ----------------------------- |
| `Byte`    | 8-bit $-2^7\sim 2^7-1$        |
| `Short`   | 16-bit $-2^{15}\sim 2^{15}-1$ |
| `Int`     | 32-bit$-2^{31}\sim 2^{31}-1$  |
| `Long`    | 64-bit$-2^{63}\sim 2^{63}-1$  |
| `Char`    | Unicode character             |
| `String`  |                               |
| `Float`   | single-precision float        |
| `Double`  | double-precision float        |
| `Boolean` |                               |

在Scala中一切操作皆方法，这意味着Scala中的一切皆为对象

与if不同的是，while与do while不能用作表达式，也即其返回值为Unit

```scala
val = forLineLengths = for{
    file <- filesHere
    if file.getName.endsWith(".scala")
    line <- fileLines(file)
    trimmed = line.trim
    if trimmed.matches(".*for.*")
} yield trimmed.length
```



##### 数组

```scala
//数组的遍历
for( i<-0 to arr.length-1)
for (i<-0 until arr.length)
for (i <- arr)
for (i <-0 until (arr.length, 2)) //步长为2
for（i <- (0 until arr.length).reverse) //倒叙
```

##### list

List与Array有着诸多的相似之处，但它们有两个明显的区别：
1 List一但创建，其值不能被改变

2 List具有递归结构（Recursive Structure),例如链表结构

List类型和其它类型集合一样，它具有协变性（Covariant)，即对于类型S和T，如果S是T的子类型，则List[S]也是List[T]的子类型

```scala
val nums = 1::2::3::4::Nil //list构造

nums.isEmpty
nums.head
nums.tail
nums.tail.head//list第二个元素
List(1,2,3):::List(4,5,6)
nums.reverse
nums.init //取除最后一个元素外的元素，返回的是list
nums zip chars // zip操作
nums.toArray//转换成数组
```

```scala
//List伴生对象方法
List.apply(1,2,3)
List.range(2,6,2)//List(2,4)
List.unzip()
List.flatten
List.concat(a, b)
```

scala中所有的集合都来自于scala.collection包及其子包mutable, immutable当中。在scala中，默认使用的都是immutable集合，如果要使用mutable集合，需要在程序中引入

scala.collection包中的集合类层次结构如下图

![](../../pictures/7.png)

scala.collection.immutable包中的类层次结构:

![](../../pictures/8.png)

scala.collection.mutable包中的类层次结构:

![](../../pictures/9.png)

##### Set操作

Set（集）是一种不存在重复元素的集合，它与数学上定义的集合是对应的

```scala
Set(3.0, 5)
```

##### Map

Map是一种键值对的集合，一般将其翻译为映射

```scala
student = Map("john"->21, "stephen"->22, "lucy"->24)
student.foreach(e=>{val (k, v) = e; print(k+":"+v)})
student.foreach(e=> println(e._1+":"+e._2))
student.contains("spark")
student.get("john")
```

##### 元组

元组则是不同类型值的聚集

```scala
val tuple=("hello", 1)
```

##### 队列

```scala
import scala.collection.immutable.Queue
val queue = Queue(1,2,3)
queue.dequeue//出队
queue.enqueue//入队
```



##### 函数与闭包

![](../../pictures/6.png)

函数字面量（function literal），也称值函数（function values），指的是函数可以赋值给变量。

```scala
/*函数字面量 =>左侧表示输入，右侧表示转换操作
*/
val increase = (x:Int) => x+1
```

```scala
//匿名函数写法
val array = Array(1,2,3,4)
val s = array.map((x:Int)=>x+1)
val s1 = array.map{(x:Int) =>x+1} //花括号方式
val s2 = array map{(x:Int) =>x+1} // 省略.的方式
val s3 = array.map((x) => x+1) //参数类型推断写法
val s4 = array.map(x=>x+1) // 函数只有一个参数的话，可以省略()
val s5 = array.map(_+1) //如果参数右边只出现一次，则可以进一步简化
```

```scala
//函数参数
def convertIntToString(f:(Int)=>String) = f(4)
//高阶函数可以产生新的函数
//(Double)=>((Double)=>)
def multiplyBy(factor:Double) = (x: Double) => factor*x

```

##### 类和对象

```scala
class Person{
    //类成员必须初始化
    var name: String = null
}
```

从字节码文件内容可以看到：虽然我们只在Person类中定义了一个类成员（域）name，类型为String，但Scala会默认帮我们生成`name()`与`name_=()`及构造函数`Person()`。其中`name()`对应java中的`getter`方法，`name_=()`对应java中的`setter`方法（由于JVM中不允许出现=，所以用$eq代替。值得注意的是定义的是公有成员，但生成的字节码中却是以私有的方式实现的，生成的getter、setter方法是公有的

```scala
class Person{
    private var privateName:String= null;
    def name = privateName
    def name_=(name:String){
        this.privateName = name
    }
}
```

从生成的字节码中可以看出：（1）定义成私有成员，其getter、setter方法也是私有的；（2）直接能访问的是我们自己定义的getter、setter方法。

如果类的成员域是val类型的变量，则只会生成getter方法

```scala
class Person{
    val name:String = "john"
}
```

从字节码文件中可以看出：val变量对应的是java中的final类型变量，只生成了getter方法

如果将成员域定义为private[this]，则不会生成getter、setter方法

```scala
class Person{
    private[this] var name: String = "john"
}
```

| Scala Field                  | Generated Methods                                            | When to Use                                                  |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `val/var name`               | `public name; name_=(var only)`                              | To implement a property that is publicly accessible and backed by a filed |
| `@BeanProperty val/var name` | `public name; getName(); name_=(var only); setName(...)(var only)` | to interoperate with JavaBeans                               |
| `private val/var name`       | `private name; name_=(var only)`                             | to confine the filed to the methods of this class. use private unless you really want a public property |
| `private[this] val/var name` | none                                                         | to confine the field to methods invoked on the same objects. Not commonly used |

```scala
puFeature.join(pinFeature, Seq("user_log_acct"), "full")
      .na.fill(0D).na.fill(0)
      .select(Utils.loadColsFromJson(pinJson, "pin").map(col): _*) //:_*,变量作为可变
```

