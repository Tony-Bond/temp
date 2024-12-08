# 输入法

```python
ZD	from collections import defaultdict
JS	from collections import Counter
```



```python
from operator import add, sub, mul
```



```python
defaultdict(int)
```

# 知识总结

- ==字典==具有相同的键和值，即使键的顺序不同，也将返回True

  ```python
  dict1 = {'a': 1, 'b': 2, 'c': 3}
  dict2 = {'b': 2, 'c': 3, 'a': 1}
  
  print(dict1 == dict2)  # True
  ```


- ```python
  ord('a') = 97
  ord('A') = 65
  ```

- ```python
  set1 = {1, 2, 3}
  set2 = {3, 4, 5}
  
  # 并集（Union）
  set1.union(set2)
  set1 | set2
  
  # 交集（Intersection）
  set1.intersection(set2)
  set1 & set2
  
  # 补集（Complement）
  set1.difference(set2)
  set1 - set2
  ```

- ```python
  float(inf)
  ```

- ```python
  " ".join(reversed(s.split()))
  ```

- ```python
  '/': lambda x, y: int(x / y)
  ```

- ``````python
  ones = s.count('1')  # 统计字符串中1的个数
  ``````

- ```python
   for j in range(i*i, n+1): 若i*i>n+1,则跳过，可以少写逻辑判断
  ```

- 



# ACM读取题目输入

[牛客算法题的输入写法记录(python版)_牛客python [1, 2, 2, 3, 3, 4, 4, 5, 5, 6\]变成1 2 2 3 3-CSDN博客](https://blog.csdn.net/watermelon12138/article/details/107367224)



# 数组

## 2分

### ==有序数组==

 <img src="./assets/1628933645-lfjMLm-image.png" alt="image.png" style="zoom:50%;" />



### [二分查找](https://leetcode.cn/problems/binary-search/)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        while l <= r:
            m = l+(r-l)//2
            t = nums[m]  #不要写错成nums(m)
            if t == target:
                return m
            elif t > target:
                r = m - 1
            else:
                l = m + 1
        return -1
```



### [搜索插入位置](https://leetcode.cn/problems/search-insert-position/)

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        while l <= r:
            m = l+(r-l)//2
            t = nums[m]
            if t >= target:  #区别
                r = m - 1
            else:
                l = m + 1
        return l
```



### [在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/)

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def sb(nums, target): #与上一题代码相同
            l, r = 0, len(nums)-1
            while l <= r:
                m = l+(r-l)//2
                t = nums[m]
                if t >= target:
                    r = m - 1
                else:
                    l = m + 1
            return l
        
        l = sb(nums, target)
        r = sb(nums, target+1) - 1

        if l == len(nums) or nums[l] != target: #条件顺序不能写反
            return [-1, -1]
        return [l, r]
```





## 移除元素

![27.移除元素-双指针法](./assets/27.移除元素-双指针法.gif)

### [移除元素](https://leetcode.cn/problems/remove-element/)

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        f = s = 0
        ln = len(nums)
        
        while f < ln:
            if nums[f] != val:   #和下一题的区别
                nums[s] = nums[f]
                s += 1
            f += 1
        return s #s代表移除后的数组长度
```



### [删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)

[动画](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/solutions/728105/shan-chu-pai-xu-shu-zu-zhong-de-zhong-fu-tudo)

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        f=s=1 #和上一题的区别
        ln = len(nums)
        
        while f < ln:
            if nums[f] != nums[f-1]:  #和上一题的区别
                nums[s] = nums[f]
                s += 1
            f += 1
        return s #s代表去重后的数组长度
```



```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        for i, x in enumerate(sorted(set(nums))): 
            nums[i] = x
        return i + 1
```



### [移动零](https://leetcode.cn/problems/move-zeroes/)

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        f = s = 0
        ln = len(nums)

        while f < ln:
            if nums[f] != 0:  #和上一题的区别
                nums[s] = nums[f]
                s += 1
            f += 1
        
        for i in range(ln-s):   #最后补0      
            nums[-1-i] = 0

```



---

上面是双指针

### [比较含退格的字符串](https://leetcode.cn/problems/backspace-string-compare/)

<img src="./assets/1603076585-eXJKxl-844.比较含退格的字符串.gif" alt="844.比较含退格的字符串.gif" style="zoom: 67%;" />

> 题目明确，注意：如果对空文本输入退格字符，文本继续为空。

```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        def build(s):
            ret = []
            for ch in s:
                if ch != "#":
                    ret.append(ch)
                elif ret:   #注意！！！！ 
                    ret.pop()
            return ret
        
        return build(s) == build(t)  
```



## 有序数组的平方

### [有序数组的平方](https://leetcode.cn/problems/squares-of-a-sorted-array/)

```Python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        return sorted([n**2 for n in nums])
```



## 长度最小的子数组

### [长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/)

![image-20240212100817337](./assets/image-20240212100817337.png)

```Python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        f=s=0
        ln = len(nums)

        ret = ln + 1
        total = 0

        while f < ln:
            total += nums[f]
            while total >= target:
                ret = min(ret, f-s+1)
                total -= nums[s]
                s += 1
            f += 1
       
        return 0 if ret == ln+1 else ret
```



---

滑窗模板

```python
# 最小滑窗模板
while j < len(nums):
    判断[i, j]是否满足条件
    while 满足条件：
        不断更新结果	#(注意在while内更新！)
        i += 1 （最大程度的压缩i，使得滑窗尽可能的小）
    j += 1

# 最大滑窗模板
while j < len(nums):
    判断[i, j]是否满足条件
    while 不满足条件：
        i += 1 （最保守的压缩i，一旦满足条件了就退出压缩i的过程，使得滑窗尽可能的大）
    不断更新结果	#（注意在while外更新！）
    j += 1
```



### <span id='jump'>[最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)</span>

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        lns,lnt = len(s),len(t)
        if lns < lnt:return ""

        mp={}
        for c in t:
            mp[c] = mp.get(c,0)+1
        lg = len(mp)

        ans = ""
        l=r=0

        while r<lns:
            while r<lns and lg:
                if s[r] in mp:
                    mp[s[r]] -= 1
                    if mp[s[r]] == 0:lg-=1
                r += 1
            if r == lns and lg:break

            while l<r and not lg:
                if s[l] in mp:
                    mp[s[l]]+=1
                    if mp[s[l]] > 0:lg+=1
            
                l += 1
            if not ans or len(ans)>r-l+1: #or
                ans = s[l-1:r]
        return ans
```



### [水果成篮](https://leetcode.cn/problems/fruit-into-baskets/)

![image-20240212105546379](./assets/image-20240212105546379.png)

```python
from collections import defaultdict

class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        # 初始化
        l=r=0
        ans = 0
        
        mp = defaultdict(int)
        lg = 0
        
        while r < len(fruits):
            if mp[fruits[r]] == 0:
                lg += 1
            
            mp[fruits[r]] += 1

            while lg > 2:
                if mp[fruits[l]] == 1:
                    lg -= 1
                mp[fruits[l]] -= 1
                l += 1

            ans = max(ans, r - l + 1)
            r += 1
        return ans
```



## 螺旋矩阵

![image-20240128213022632](./assets/image-20240128213022632.png)

### [螺旋遍历二维数组](https://leetcode.cn/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

```python
class Solution:
    def spiralArray(self, matrix: List[List[int]]) -> List[int]:
        if not matrix: return []
    
        n = len(matrix)
        m = len(matrix[0])
        top, bottom = 0, n - 1
        left, right = 0, m - 1
        
        res = []
        
        while True:
            # 左 -> 右
            for i in range(left, right + 1):
                res.append(matrix[top][i])
            top += 1
            if top > bottom: break

            # 上 -> 下
            for i in range(top, bottom + 1):
                res.append(matrix[i][right])
            right -= 1
            if left > right: break

            # 右 -> 左
            for i in range(right, left - 1, -1):
                res.append(matrix[bottom][i])
            bottom -= 1
            if top > bottom: break

            # 下 -> 上
            for i in range(bottom, top - 1, -1):
                res.append(matrix[i][left])
            left += 1
            if left > right: break
            
        return res
```



### [螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)

【解题同上】

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        l,r = 0,len(matrix[0])-1
        t,b = 0,len(matrix)-1

        ans = []
        while True:
            for i in range(l,r+1):
                ans.append(matrix[t][i])
            t += 1
            if t > b:break

            for i in range(t,b+1):
                ans.append(matrix[i][r])
            r -= 1
            if l > r:break

            for i in range(r, l-1, -1):
                ans.append(matrix[b][i])
            b -= 1
            if t > b: break

            for i in range(b, t-1, -1):
                ans.append(matrix[i][l])
            l += 1
            if l > r:break
        return ans
```



### [螺旋矩阵 II](https://leetcode.cn/problems/spiral-matrix-ii/)

```python
class Solution:
    def generateMatrix(self, n: int) -> [[int]]:
        l=t=0
        r=b=n-1
        
        mat = [[0 for _ in range(n)] for _ in range(n)]
        num, tar = 1, n * n

        while num <= tar:
            for i in range(l, r + 1): # left to right
                mat[t][i] = num
                num += 1
            t += 1
            for i in range(t, b + 1): # top to bottom
                mat[i][r] = num
                num += 1
            r -= 1
            for i in range(r, l - 1, -1): # right to left
                mat[b][i] = num
                num += 1
            b -= 1
            for i in range(b, t - 1, -1): # bottom to top
                mat[i][l] = num
                num += 1
            l += 1
        return mat

# 法2 写法同上题
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        l=t=0
        r=b=n-1

        mat=[[0 for _ in range(n)] for _ in range(n)]

        num=1

        while True:
            for i in range(l,r+1):
                mat[t][i]=num
                num+=1
            t += 1
            if t > b:break

            for i in range(t,b+1):
                mat[i][r]=num
                num+=1
            r -= 1
            if l > r:break

            for i in range(r,l-1,-1):
                mat[b][i]=num
                num+=1
            b -=1
            if t > b:break

            for i in range(b,t-1,-1):
                mat[i][l]=num
                num+=1
            l += 1
            if l > r:break
        return mat
```



# 链表

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```



## [移除链表元素](https://leetcode.cn/problems/remove-linked-list-elements/)

```python
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy = ListNode(next = head)  
        cur = dummy
        
        while cur.next:
            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        
        return dummy.next
```



## [设计链表](https://leetcode.cn/problems/design-linked-list/)[背]

> self.dummy 的index为 -1
>
> self.dummy.next，也就是head 的index为 0
>
> ```python
> cur = dummy
> for i in range(index+1):
> 	cur = cur.next
> return cur.val
> ```

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class MyLinkedList:
    def __init__(self):
        self.dummy = ListNode()
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        
        cur = self.dummy.next
        for i in range(index):
            cur = cur.next
            
        return cur.val
	
    # 接链表头
    def addAtHead(self, val: int) -> None:
        self.dummy.next = ListNode(val, self.dummy.next)
        self.size += 1
	
    # 接链表尾巴
    def addAtTail(self, val: int) -> None:
        cur = self.dummy
        while cur.next:
            cur = cur.next
        cur.next = ListNode(val)
        self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return
        
        cur = self.dummy
        for i in range(index):
            cur = cur.next
        cur.next = ListNode(val, cur.next)
        self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        
        cur = self.dummy
        for i in range(index):
            cur = cur.next
        cur.next = cur.next.next
        self.size -= 1
```



## [反转链表](https://leetcode.cn/problems/reverse-linked-list/)[无dummy]

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur, pre = head, None
        while cur:
            tmp = cur.next # 暂存后继节点 cur.next
            cur.next = pre # 修改 next 引用指向
            pre = cur      # pre 暂存 cur
            cur = tmp      # cur 访问下一节点
        return pre
```



## [两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/)

<img src="./assets/image-20240129121015370.png" alt="image-20240129121015370" style="zoom:150%;" />

```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = ListNode(next=head)
        c = dummy
        while c.next and c.next.next:
            a, b = c.next, c.next.next
            c.next, a.next = b, b.next
            b.next = a
            c = c.next.next
        return dummy.next
```



## [删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0, head)
        
        slow = fast = dummy        
        
        for i in range(n+1): # n+1 步
            fast = fast.next
        
        while fast:
            slow = slow.next
            fast = fast.next
        
        slow.next = slow.next.next
        
        return dummy.next
```



## <span id='jump1'>[链表相交](https://leetcode.cn/problems/intersection-of-two-linked-lists-lcci/)</span>

![image-20240116192546101](./assets/image-20240116192546101.png)

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None
                    
        A, B = headA, headB
        # AB若有交点，返回交点；没有交点，刚好链尾返回None
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A
```



## [环形链表](https://leetcode.cn/problems/linked-list-cycle/)[背]

```python
class Solution:
    def hasCycle(self, head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```



## [环形链表 II](https://leetcode.cn/problems/c32eOV/)

<img src="./assets/1630857284-xUemfr-image.png" alt="image.png" style="zoom:33%;" />

假设快慢指针在W点相遇，此时慢指针走过的路程为x+y，快指针走过的路程为x+y+n(y+z)。
为什么慢指针走过y就必然与快指针相遇，而不是慢指针走过y+m(y+z)呢？

首先，由于快指针一定先进入环内，这点毋庸置疑。
而且，快指针是慢指针速度的二倍，即慢指针走完一圈，快指针可以走两圈
所以不论慢指针入环时，快指针在哪一点，快指针都可以在慢指针未走过一圈时追上慢指针。
而由于快指针走过的节点数是慢指针的二倍，所以得到公式：
(x + y) * 2 = x + y + n (y + z)
两边抵消 x+y，得到 x + y = n (y + z)
由于我们最终要求的是x，所以 x = n (y + z) - y
然而,此时慢指针所走过的路程刚好为y，如果此时有一个指针point从头开始走向环，即x路程
那么，慢指针刚好要走过的就是 n (y + z) - y + y = n (y + z)
即 point 走x的距离到达环的入口的时刻，刚好为slow走过n圈到达入口，两个指针相遇？
得到这个结论，那么题目就迎刃而解了！

```python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        f=s=head
        while f and f.next:
            f=f.next.next
            s=s.next
            if f==s:   # 将上一题改写
                point=head
                while point != s:
                    point=point.next
                    s=s.next
                return point
        return None 
```



# 哈希表

==字典==具有相同的键和值，即使键的顺序不同，也将返回True

```python
dict1 = {'a': 1, 'b': 2, 'c': 3}
dict2 = {'b': 2, 'c': 3, 'a': 1}

print(dict1 == dict2)  # True
```



## 有效的字母异位词

### [有效的字母异位词](https://leetcode.cn/problems/valid-anagram/)

[类似题目](#jump)

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s)!=len(t):return False
        def build(s):
            mp={}
            for c in s:
                mp[c]=mp.get(c,0)+1
            return mp

        s = build(s)
        t = build(t)

        return s==t
 
# 法二
 class Solution(object):
    def isAnagram(self, s: str, t: str) -> bool:
        from collections import Counter
        s = Counter(s)
        t = Counter(t)
        return s==t
```



### [字母异位词分组](https://leetcode.cn/problems/group-anagrams/)

![image-20240129111659178](./assets/image-20240129111659178.png)

```python
from collections import defaultdict

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = collections.defaultdict(list)

        for st in strs:
            key = "".join(sorted(st))
            mp[key].append(st)
        
        return list(mp.values())

#法二  
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = collections.defaultdict(list)

        for st in strs:
            counts = [0] * 26
            for ch in st:
                counts[ord(ch) - ord("a")] += 1
            # 需要将 list 转换成 tuple 才能进行哈希
            mp[tuple(counts)].append(st)
        
        return list(mp.values())
```

https://leetcode.cn/problems/find-all-anagrams-in-a-string/solutions/9749/hua-dong-chuang-kou-tong-yong-si-xiang-jie-jue-zi-/ 



### [找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)

<img src="./assets/image-20240129122300141.png" alt="image-20240129122300141" style="zoom:120%;" />

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        s_len, p_len = len(s), len(p)
        
        if s_len < p_len:
            return []
		
        ans = []
        s_count = [0] * 26
        p_count = [0] * 26
        
        for i in range(p_len):
            s_count[ord(s[i]) - 97] += 1
            p_count[ord(p[i]) - 97] += 1
		
        if s_count == p_count:
            ans.append(0)
		
        for i in range(s_len - p_len):
            s_count[ord(s[i]) - 97] -= 1
            s_count[ord(s[i + p_len]) - 97] += 1
            	
            if s_count == p_count:
                ans.append(i + 1)
		
        return ans
```



## [两个数组的交集](https://leetcode.cn/problems/intersection-of-two-arrays/)

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        mp={}
        for i in nums1:
            mp[i]=mp.get(i,0) + 1
        
        ans={}

        for i in nums2:
            if i in mp:
                ans[i]=ans.get(i,0)+1
        return list(ans.keys())

# 法二
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        set1 = set(nums1)
        set2 = set(nums2)
        return self.set_intersection(set1, set2)

    def set_intersection(self, set1, set2):
        if len(set1) > len(set2):
            return self.set_intersection(set2, set1)
        return [x for x in set1 if x in set2]
    
# 法三
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return list(set(nums1) & set(nums2))    # 两个数组先变成集合，求交集后还原为数组

# 法四
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    # 使用哈希表存储一个数组中的所有元素
        table = {}
        for num in nums1:
            table[num] = table.get(num, 0) + 1
        
        # 使用集合存储结果
        res = set()
        for num in nums2:
            if num in table:
                res.add(num)
                del table[num]
        
        return list(res)
```



## [快乐数](https://leetcode.cn/problems/happy-number/)

![image-20240129130858008](./assets/image-20240129130858008.png)

```python
class Solution:
   def isHappy(self, n: int) -> bool:
       seen = set()
       while n != 1:
           n = sum(int(i) ** 2 for i in str(n))
           if n in seen:
               return False
           seen.add(n)
       return True
```



## [四数相加 II](https://leetcode.cn/problems/4sum-ii/)

```python
from collections import Counter
class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        countAB = collections.Counter(u + v for u in A for v in B)
        ans = 0
        for u in C:
            for v in D:
                if -u - v in countAB:
                    ans += countAB[-u - v]
        return ans
```



## [最小操作次数使数组元素相等](https://leetcode.cn/problems/minimum-moves-to-equal-array-elements/)

逆向思考：其中一个数减1

```python
class Solution:
    def minMoves(self, nums: List[int]) -> int:
        min_num = min(nums)
        res = 0
        for num in nums:
            res += num - min_num
        return res
```



## [三数之和](https://leetcode.cn/problems/3sum/)

```python
class Solution:
    def threeSum(self, li: List[int]) -> List[List[int]]:
        li.sort()
        ans = []
        for i in range(len(li)-2):
            if li[i] > 0:break
            if i>0 and li[i]==li[i-1]:continue
            l,r = i+1,len(li)-1
            while l<r:
                s = li[i]+li[l]+li[r]
                if s<0:
                    l += 1
                    while l<r and li[l] == li[l-1]:l += 1
                elif s>0:
                    r -= 1
                    while l<r and li[r] == li[r+1]:r -= 1
                else:
                    ans.append([li[i],li[l],li[r]])
                    l+=1
                    r-=1
                    while l<r and li[l] == li[l-1]:l += 1
                    while l<r and li[r] == li[r+1]:r -= 1
        return ans
```



## [四数之和](https://leetcode.cn/problems/4sum/)

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        ans = []
        for i in range(len(nums)-3):
            # if nums[i]>0:break
            if i>0 and nums[i]==nums[i-1]:continue

            for j in range(i+1, len(nums)-2):
                if j>i+1 and nums[j]==nums[j-1]:continue
                
                l,r=j+1,len(nums)-1
                while l<r:
                    tot = nums[i]+nums[j]+nums[l]+nums[r]
                    if tot==target:
                        ans.append([nums[i],nums[j],nums[l],nums[r]])
                        l+=1
                        r-=1
                        while l<r and nums[l]==nums[l-1]:
                            l+=1
                        while l<r and nums[r]==nums[r+1]:
                            r-=1
                    elif tot>target:
                        r-=1
                        while l<r and nums[r]==nums[r+1]:
                            r-=1
                    else:
                        l+=1
                        while l<r and nums[l]==nums[l-1]:
                            l+=1
        return ans
```



# 字符串

## [反转字符串](https://leetcode.cn/problems/reverse-string/)

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        l,r=0,len(s)-1
        while l<r:
            s[l],s[r]=s[r],s[l]
            l+=1
            r-=1
```



## [反转字符串 II](https://leetcode.cn/problems/reverse-string-ii/)

```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        # Two pointers. Another is inside the loop.
        p = 0
        while p < len(s):
            p2 = p + k
            # Written in this could be more pythonic.
            s = s[:p] + s[p: p2][::-1] + s[p2:]
            p = p + 2 * k
        return s

```



## [反转字符串中的单词](https://leetcode.cn/problems/reverse-words-in-a-string/)

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        return " ".join(reversed(s.split()))
    
# 法2    
class Solution:
    def reverseWords(self, s: str) -> str:
        ls = s.split()
        return " ".join(ls[::-1])
```



## [重复的子字符串](https://leetcode.cn/problems/repeated-substring-pattern/)

[思路](https://leetcode.cn/problems/repeated-substring-pattern/solutions/386644/gou-zao-shuang-bei-zi-fu-chuan-by-elevenxx)

```python
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        return s in (s+s)[1:-1] 
    
# 法2
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        return (s + s).find(s, 1) < len(s)
```





# 栈与队列

![栈与队列理论1](./assets/20210104235346563.png)

队列是先进先出，栈是先进后出

## [用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/)

```python
class MyQueue(object):

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x):
        self.stack1.append(x)

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        """
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()

    def peek(self):
        """
        Get the front element.
        """
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1]

    def empty(self):
        return not self.stack1 and not self.stack2


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```



## [用队列实现栈](https://leetcode.cn/problems/implement-stack-using-queues/)

```python
from collections import deque
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.q1 = deque()
        self.q2 = deque()


    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1


    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        return self.q1.popleft()


    def top(self) -> int:
        """
        Get the top element.
        """
        return self.q1[0]


    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return not self.q1


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
```



## [有效的括号](https://leetcode.cn/problems/valid-parentheses/)

```python
class Solution:
    def isValid(self, s: str) -> bool:
        if len(s) % 2 == 1:
            return False
        
        pairs = {
            ")": "(",
            "]": "[",
            "}": "{",
        }
        stack = []
        for ch in s:
            if ch in pairs:
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()
            else:
                stack.append(ch)
        
        return not stack
```



## [删除字符串中的所有相邻重复项](https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/)

```python
class Solution:
    def removeDuplicates(self, s: str) -> str:
        ls = ['1']
        
        for c in s:
            if c == ls[-1]:
                ls.pop()
            else:
                ls.append(c)
        return ''.join(ls[1:])
```



## [逆波兰表达式求值](https://leetcode.cn/problems/evaluate-reverse-polish-notation/)

```PYTHON
from operator import add, sub, mul

class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        mp = {'+': add, '-': sub, '*': mul, '/': lambda x, y: int(x / y)}        
        stk = []
        for t in tokens:
            if t in mp:
                a,b=stk.pop(),stk.pop()
                stk.append(mp[t](b,a)) #注意a,b顺序
            else:
                stk.append(int(t))
        return stk.pop()
    
#关于运算，另一种处理方法
def evaluate(self, num1, num2, op):
    if op == "+":
        return num1 + num2
    elif op == "-":
        return num1 - num2
    elif op == "*":
        return num1 * num2
    elif op == "/":
        return int(num1 / float(num2))
```



## [前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/)

```python
```



## [滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/)

[[题解]](https://leetcode.cn/problems/sliding-window-maximum/solutions/)

> 遍历数组，将 数 存放在双向队列中，并用 L,R 来标记窗口的左边界和右边界。
> 队列中保存的并不是真的 数，而是该数值对应的数组下标位置，并且数组中的数要从大到小排序。
> 如果当前遍历的数比队尾的值大，则需要弹出队尾值，直到队列重新满足从大到小的要求。
> 刚开始遍历时，L 和 R 都为 0，有一个形成窗口的过程，此过程没有最大值，L 不动，R 向右移。当窗口大小形成时，L 和 R 一起向右移，每次移动时，判断队首的值的数组下标是否在 [L,R] 中，如果不在则需要弹出队首的值，当前窗口的最大值即为队首的数。

![image-20240130132740720](./assets/image-20240130132740720.png)

```python
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums or len(nums) < 2: 
            return nums

        que = deque()
        res = []

        for i, n in enumerate(nums):
            # 保证从大到小 如果前面数小则需要依次弹出，直至满足要求
            while que and nums[que[-1]] <= n:
                que.pop()

            # nums中元素的下标，加入队列中   
            que.append(i)

            # 判断队首值是否有效
            if que[0] <= i - k:
                que.popleft()
            
            # 当窗口长度为k时 保存当前窗口中最大值
            if i + 1 >= k:
                res.append(nums[que[0]])
        return res
```



<div style="page-break-after:always;"></div>

# 二叉树

```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
```



## 递归遍历(DFS/栈)

### [前序遍历](https://leetcode.cn/problems/binary-tree-preorder-traversal/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        left = self.preorderTraversal(root.left)
        right = self.preorderTraversal(root.right)

        return  [root.val] + left +  right
```

### [中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []

        left = self.inorderTraversal(root.left)
        right = self.inorderTraversal(root.right)

        return left + [root.val] + right
```

### <span id='djjs'>[后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/)</span>

```python
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []

        left = self.postorderTraversal(root.left)
        right = self.postorderTraversal(root.right)

        return left + right + [root.val]
```



## 迭代遍历

https://leetcode.cn/problems/binary-tree-preorder-traversal/solutions/247053/tu-jie-er-cha-shu-de-si-chong-bian-li-by-z1m

### [前序遍历](https://leetcode.cn/problems/binary-tree-preorder-traversal/)

```python
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:return []

        cur = root
        stk =[]
        ans = []

        while cur or stk:
            while cur:
                ans.append(cur.val)
                stk.append(cur)
                cur=cur.left
            tmp = stk.pop()
            cur = tmp.right
        return ans
```

### [中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root: return []

        cur =root
        stk=[]
        ans =[]

        while stk or cur:
            while cur:
                stk.append(cur)
                cur = cur.left
            tmp=stk.pop()
            cur = tmp.right
            ans.append(tmp.val)
        return ans
```

### [后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/)

```python
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:return []

        cur = root
        stk = []
        ans = []

        while stk or cur:
            while cur:
                ans.append(cur.val)
                stk.append(cur)
                cur=cur.right
            tmp=stk.pop()
            cur=tmp.left
        return ans[::-1]
```



## [层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)(BFS/队列)

> [扩展](https://leetcode.cn/problems/binary-tree-level-order-traversal/solutions/244853/bfs-de-shi-yong-chang-jing-zong-jie-ceng-xu-bian-l/)
>
> 很多同学一看到「最短路径」，就条件反射地想到「Dijkstra 算法」。为什么 BFS 遍历也能找到最短路径呢？
>
> 这是因为，「Dijkstra 算法」解决的是带权最短路径问题，而我们这里关注的是无权最短路径问题。
>
> 也可以看成每条边的权重都是 1。这样的最短路径问题，用 BFS 求解就行了。
>
> 在面试中，你可能更希望写 BFS 而不是 Dijkstra。毕竟，敢保证自己能写对 Dijkstra 算法的人不多。

[图解](https://leetcode.cn/problems/binary-tree-level-order-traversal/solutions/2361604/102-er-cha-shu-de-ceng-xu-bian-li-yan-du-dyf7)

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        q = collections.deque([root])
        ans = []
        
        while q:
            level = []
            for _ in range(len(q)):
                cur = q.popleft()
                level.append(cur.val)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            ans.append(level)
        return ans
```



## [N 叉树的层序遍历](https://leetcode.cn/problems/n-ary-tree-level-order-traversal/)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root:return []

        from collections import deque

        q = deque([root])
        ans=[]

        while q:
            level=[]
            for _ in range(len(q)):
                tmp = q.popleft()
                level.append(tmp.val)
                
                for child in tmp.children: #添加逻辑判断
                    q.append(child)
                    
            ans.append(level)
        return ans
```



## [填充每个节点的下一个右侧节点指针](https://leetcode.cn/problems/populating-next-right-pointers-in-each-node/)

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root
        
        q = collections.deque([root])
        
        while q:
            level_size = len(q)
            prev = None
            
            for i in range(level_size):
                node = q.popleft()
                
                if prev: #添加逻辑判断
                    prev.next = node
                
                prev = node
                
                if node.left:
                    q.append(node.left)
                
                if node.right:
                    q.append(node.right)
        
        return root
```



## [二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

```python
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        depth = 0
        q = collections.deque([root])
        
        while q:
            
            depth += 1
            
            for _ in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        
        return depth
```



## [二叉树的最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/)

```python
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        depth = 0
        q = collections.deque([root])
        
        while q:
            depth += 1 
            for _ in range(len(q)):
                node = q.popleft()
                
                if not node.left and not node.right: #多一个判断逻辑
                    return depth
            
                if node.left:
                    q.append(node.left)
                    
                if node.right:
                    q.append(node.right)

        return depth
```



## [翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/)

```python
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root: 
            return None

        q = collections.deque([root])
        
        while q:
            for i in range(len(q)):
                node = q.popleft()
                
                node.left, node.right = node.right, node.left
                
                if node.left: 
                    q.append(node.left)
                if node.right: 
                    q.append(node.right)
        return root   
```



## [对称二叉树](https://leetcode.cn/problems/symmetric-tree/)

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        q = collections.deque([root.left, root.right]) #
        
        while q:
            ln = len(q)            
            if ln % 2 != 0:
                return False
            
            level = []
            for i in range(ln):
                node = q.popleft()
                
                if node:
                    level.append(node.val)
                    q.append(node.left)
                    q.append(node.right)
                else:
                    level.append(None)
                    
            if level != level[::-1]:
                return False
            
        return True
```



## [相同的树](https://leetcode.cn/problems/same-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        queue = [p, q]

        while queue:
            pNode = queue.pop(0)
            qNode = queue.pop(0)

            if not pNode and not qNode:
                continue
            
            if not pNode or not qNode:
                return False
            
            if pNode.val != qNode.val:
                return False

            queue.append(pNode.left)
            queue.append(qNode.left)

            queue.append(pNode.right)
            queue.append(qNode.right)

        return True
```



## [另一棵树的子树](https://leetcode.cn/problems/subtree-of-another-tree/) 

结合上一题

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:      
        qe=[p,q]

        while qe:
            pn = qe.pop(0)
            qn = qe.pop(0)

            if not pn and not qn:
                continue
            
            if not pn or not qn:return False

            if pn.val != qn.val:return False

            qe.append(pn.left)
            qe.append(qn.left)

            qe.append(pn.right)
            qe.append(qn.right)
        return True

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        q = [root]

        while q:
            r = q.pop(0)

            if r.val==subRoot.val:
                if self.isSameTree(r,subRoot):
                    return True
            
            if r.left:q.append(r.left)
            if r.right:q.append(r.right)
        return False
```



## [平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/)

[解题](https://leetcode.cn/problems/balanced-binary-tree/solutions/2099334/tu-jie-leetcode-ping-heng-er-cha-shu-di-gogyi/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root): # 求二叉树最大深度题
        if not root:
            return 0
        
        depth = 0
        q = collections.deque([root])
        
        while q:            
            depth += 1            
            for _ in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)        
        return depth

    def isBalanced(self, root: TreeNode) -> bool: # 中序遍历题
        if root == None:
            return True

        cur =root
        stk=[]

        while stk or cur:
            while cur:
                stk.append(cur)
                cur = cur.left
            tmp=stk.pop()
            
            if(abs(self.maxDepth(tmp.left) - self.maxDepth(tmp.right)) > 1): #添加的新逻辑
                    return False
                
            cur = tmp.right                      
        return True
```



## [二叉树的所有路径](https://leetcode.cn/problems/binary-tree-paths/)

https://leetcode.cn/problems/binary-tree-paths/solutions/1366444/acm-xuia-by-rocky0429-2-ul6r/

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        if not root:
            return []

        stack = [root]
        pth = [str(root.val)]
        ans = []

        while stack:
            node = stack.pop()
            path = pth.pop()
            # 如果当前节点为叶子节点
            if not node.left and not node.right:
                ans.append(path)
            
            # 右子树入栈
            if node.right:
                stack.append(node.right)
                pth.append(path + "->" + str(node.right.val))
            
            # 左子树入栈
            if node.left:
                stack.append(node.left)
                pth.append(path + "->" + str(node.left.val))

        return ans
```



## [左叶子之和](https://leetcode.cn/problems/sum-of-left-leaves/)

```python
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        
        stk = [root]
        ans = 0

        while stk:
            node = stk.pop()            
            if node.left:
                if not node.left.left and not node.left.right:
                    ans+=node.left.val
                stk.append(node.left)
            if node.right:
                stk.append(node.right)
        return ans
```



## [找树左下角的值](https://leetcode.cn/problems/find-bottom-left-tree-value/)

重点：层序遍历，保存每一层的第一个元素

```python
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        q = [root]
        ans = 0

        while q:
            n = len(q)
            for i in range(n):           
            	# 存储每一层的第一个元素
                if i == 0:
                    ans = q[i].val
                node = q.pop(0)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)        
        return ans
```



## [路径总和](https://leetcode.cn/problems/path-sum/)

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:return False
        
        stk = [root]
        pth = [root.val]
        
        while stk:
            tmp = stk.pop()
            path = pth.pop()

            if not tmp.left and not tmp.right:
                if path == targetSum:
                    return True

            if tmp.left:
                stk.append(tmp.left)
                pth.append(path+tmp.left.val)
            if tmp.right:
                stk.append(tmp.right)
                pth.append(path+tmp.right.val)
        return False
```



## [路径总和 II](https://leetcode.cn/problems/path-sum-ii/)

```python
from collections import defaultdict
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        if not root:return []
        
        stk = [root]
        pth = [[root.val]]
        ans = []

        while stk:
            tmp = stk.pop()
            p = pth.pop()

            if not tmp.left and not tmp.right:
                if targetSum==sum(p):
                    ans.append(p)
            
            if tmp.left:
                stk.append(tmp.left)
                pth.append(p+[tmp.left.val])
            if tmp.right:
                stk.append(tmp.right)
                pth.append(p+[tmp.right.val])
        print(pth)
        return ans
```



## 构造二叉树

### [从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

题目有限制：`inorder` 和 `postorder` 都由 **不同** 的值组成

【技巧】中序遍历inorder定长度

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        # 递归中止条件：树为空
        if not inorder or not postorder:
            return None

        # 根节点的值为后序遍历的最后一个元素值
        root = TreeNode(postorder[-1])

        # 用根节点的值去中序数组中查找对应元素下标
        idx = inorder.index(rootVal)

        # 找出中序遍历的左子树和右子树
        l = inorder[:idx]
        r = inorder[idx + 1:]

        # 找出后序遍历的左子树和右子树
        pl = postorder[: len(l)]
        pr = postorder[len(l): len(inorder) - 1]

        root.left = self.buildTree(l, pl)
        root.right = self.buildTree(r, pr)

        return root
```



### [从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])

        idx = inorder.index(preorder[0])

        l= inorder[:idx]
        r= inorder[idx+1:]

        pl= preorder[1:len(l)+1]
        pr= preorder[len(l)+1:]

        root.left=self.buildTree(pl,l)
        root.right=self.buildTree(pr,r)
        return root
```



### [最大二叉树](https://leetcode.cn/problems/maximum-binary-tree/)

```python
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        
        root = TreeNode(max(nums))
        idx = nums.index(max(nums))

        l = nums[:idx]
        r = nums[idx+1:]

        root.left = self.constructMaximumBinaryTree(l)
        root.right = self.constructMaximumBinaryTree(r)

        return root
```



### [合并二叉树](https://leetcode.cn/problems/merge-two-binary-trees/)

```python
class Solution:
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:

        #  但凡有一个节点为空, 就立刻返回另外一个. 如果另外一个也为None就直接返回None. 
        if not root1: 
            return root2
        if not root2: 
            return root1
        # 上面的递归终止条件保证了代码执行到这里root1, root2都非空. 
        
        root1.val += root2.val # 中
        root1.left = self.mergeTrees(root1.left, root2.left) #左
        root1.right = self.mergeTrees(root1.right, root2.right) # 右
        
        return root1 # ⚠️ 注意: 本题我们重复使用了题目给出的节点而不是创建新节点. 节省时间, 空间. 

```



## 二叉搜索树

### [二叉搜索树中的插入操作](https://leetcode.cn/problems/insert-into-a-binary-search-tree/)

```python
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:return TreeNode(val)
        cur = root 
        while cur:
            v = cur.val
            if v > val:
                if cur.left:
                    cur = cur.left
                else:
                    cur.left = TreeNode(val)
                    return root
            else:
                if cur.right:
                    cur = cur.right
                else:
                    cur.right = TreeNode(val)
                    return root
```



### [删除二叉搜索树中的节点](https://leetcode.cn/problems/delete-node-in-a-bst/)

> 目标节点大于当前节点值，则去右子树中删除；
> 目标节点小于当前节点值，则去左子树中删除；
> 目标节点就是当前节点，分为以下三种情况：
>
> - 其无左子：其右子顶替其位置，删除了该节点； 
>
> - 其无右子：其左子顶替其位置，删除了该节点；
>
> - 其左右子节点都有：其==左==子树--->右子树的最左节点的左子树上，然后右子树顶替其位置，由此删除了该节点

```python
class Solution:
    def deleteNode(self, root, key):
        if not root:
            return root

        if root.val == key:
            if not root.left and not root.right:
                return None
            elif not root.left:  #无左子
                return root.right
            elif not root.right: #无右子
                return root.left
            else: #左右子节点都有
                cur = root.right
                while cur.left:
                    cur = cur.left
                cur.left = root.left
                return root.right   #别忘写
                
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        if root.val < key:
            root.right = self.deleteNode(root.right, key)
        
        return root
```



### [修剪二叉搜索树](https://leetcode.cn/problems/trim-a-binary-search-tree/)

```python
class Solution:
    def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
        if not root:
            return None
        if root.val < low:
            return self.trimBST(root.right, low, high)
        if root.val > high:
            return self.trimBST(root.left, low, high)
            
        root.left = self.trimBST(root.left, low, high)  # root.left 接入符合条件的左孩子
        root.right = self.trimBST(root.right, low, high)  # root.right 接入符合条件的右孩子
        return root
```



### [将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)

```python
class Solution:
    def traversal(self, nums: List[int], left: int, right: int) -> TreeNode:
        if left > right:
            return None
        
        mid = left + (right - left) // 2
        root = TreeNode(nums[mid])
        root.left = self.traversal(nums, left, mid - 1)
        root.right = self.traversal(nums, mid + 1, right)
        return root
    
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        root = self.traversal(nums, 0, len(nums) - 1)
        return root
```



### [把二叉搜索树转换为累加树](https://leetcode.cn/problems/convert-bst-to-greater-tree/)

> 其实这就是一棵树，换一个角度来看，这就是一个有序数组[2, 5, 13]，求从后到前的累加数组，也就是[20, 18, 13]，是不是感觉这就简单了。
>
> **反中序遍历这个二叉树，然后顺序累加就可以了**。

```python
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: 
            return root

        stack = []
        cur = root
        pre = 0

        while cur or stack:
            while cur:
                stack.append(cur)
                cur = cur.right            
            tmp = stack.pop()
            tmp.val += pre
            pre = tmp.val           
            cur =tmp.left
        return root
```



### [二叉搜索树中的搜索](https://leetcode.cn/problems/search-in-a-binary-search-tree/)

```python
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if root == None:
            return None

        while root:
            if root.val == val:
                return root
            elif root.val > val:
                root = root.left
            else:
                root = root.right
```



### [验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)

搜索树，用==中序遍历==，遍历的数值是从小到大

【递归】

```python
class Solution:
    # 中序遍历, 数值保存在res中
    def inOrder(self, root: TreeNode, res):
        if root == None:
            return
        self.inOrder(root.left, res)
        res.append(root.val)
        self.inOrder(root.right, res)


    def isValidBST(self, root: TreeNode) -> bool:
        res = []
        self.inOrder(root, res)
        # 判断 res 是否有序
        for i in range(1, len(res)):
            if num[i] <= nums[i - 1]:
                return False
        return True
```

【迭代】

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        stack = []
        pre = None

        while root or stack:
            if root:
                stack.append(root)
                root = root.left
            else:
                cur = stack.pop()
                # 判断序列是否有序
                if pre and cur.val <= pre.val:
                    return False
                pre = cur
                root = cur.right

        return True
```



### [二叉搜索树的最小绝对差](https://leetcode.cn/problems/minimum-absolute-difference-in-bst/)

【递归】

```python
class Solution:
    # 中序遍历
    def inOrder(self, root: TreeNode, res):
        if root == None:
            return
        self.inOrder(root.left, res)
        res.append(root.val)
        self.inOrder(root.right, res)

    def getMinimumDifference(self, root: TreeNode) -> int:
        res = []
        ans = float('inf')
        self.inOrder(root, res)

        for i in range(len(res) - 1):
            ans = min(abs(res[i] - res[i + 1]), ans)

        return ans
```

【非递归】

```python
class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        stk = []
        ans = float('inf')
        pre = None
 
        while root or stk:
            if root:
                stk.append(root)
                root = root.left
            else:
                cur = stk.pop()
                if pre:
                    ans = min(cur.val - pre.val, ans)
                pre = cur
                root = cur.right
        return ans
```



```python
class Solution:
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        stk = []

        mp = {}
        ans =[]
        while root or stk:
            if root:
                stk.append(root)
                root = root.left
            else:
                cur = stk.pop()
                mp[cur.val] = mp.get(cur.val,0)+1 
                root = cur.right
        
        m= max(mp.values())
        for k,v in mp.items():
            if v == m:
                ans.append(k)
        return ans
```



### [二叉搜索树中的众数](https://leetcode.cn/problems/find-mode-in-binary-search-tree/)

```python
class Solution:
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        stk = []

        mp = {}
        ans =[]
        while root or stk:
            if root:
                stk.append(root)
                root = root.left
            else:
                cur = stk.pop()
                mp[cur.val] = mp.get(cur.val,0)+1 
                root = cur.right
        
        m= max(mp.values())
        for k,v in mp.items():
            if v == m:
                ans.append(k)
        return ans
```



## [二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

> [思路](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/solutions/6152/shi-yong-zi-dian-cun-chu-shuang-qin-jie-dian-by-mi/)
>
> 由于每个节点只有唯一的父节点，可以使用到字典的value-key的形式（节点-父节点）
> 字典中预置根节点的父节点为None。
>
> 字典建立完成后，**二叉树就可以看成一个所有节点都将最终指向根节点的链表了**。
>
> 于是在二叉树中寻找两个节点的最小公共节点就相当于，在一个链表中寻找他们相遇的节点
>
> 后面的思路可以参考题目[相交链表](#jump1)

> 【扩展】将树写成链表代码
>
> ```python
> dic = {root:None}
> def dfs(node):
>     if node:
>         if node.left: 
>             dic[node.left] = node
>         if node.right: 
>             dic[node.right] = node
>         dfs(node.left)
>         dfs(node.right)
> dfs(root)
> ```



```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        dic = {root:None}
        def dfs(node):
            if node:
                if node.left: 
                    dic[node.left] = node
                if node.right: 
                    dic[node.right] = node
                dfs(node.left)
                dfs(node.right)
        dfs(root)
        l1, l2 = p, q
        while(l1!=l2):
            l1 = dic.get(l1, q)
            l2 = dic.get(l2, p)
        return l1
```



## [二叉搜索树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/)

比上一题简单，可直接用上一题解法

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        x = root.val
        if p.val < x and q.val < x:
            return self.lowestCommonAncestor(root.left, p, q)
        if p.val > x and q.val > x:
            return self.lowestCommonAncestor(root.right, p, q)
        return root
```



# 回溯算法

回溯法标准框架：

```python
def backtrack(path, selected):
    if 满足停止条件：
        res.append(path)
    for 选择 in 选择列表：
        做出选择
        递归执行backtrack
        撤销选择
```

---

回溯是递归的副产品，只要有递归就会有回溯。

**所以以下讲解中，回溯函数也就是递归函数，指的都是一个函数**。

**回溯的本质是穷举，穷举所有可能，然后选出我们想要的答案**，如果想让回溯法高效一些，可以加一些剪枝的操作，但也改不了回溯法就是穷举的本质。

---

回溯法，一般可以解决如下几种问题：

- 组合问题：N个数里面按一定规则找出k个数的集合
- 排列问题：N个数按一定规则全排列，有几种排列方式
- 切割问题：一个字符串按一定规则有几种切割方式
- 子集问题：一个N个数的集合里有多少符合条件的子集
- 棋盘问题：N皇后，解数独等等

**组合是不强调元素顺序的，排列是强调元素顺序**

组合无序，排列有序

---

```python
if not pth or nums[i]>=pth[-1]: # 需满足递增
    pth.append(nums[i])         # 选nums[i]
    bt(i+1, pth)
    pth.pop()                   # 回溯复原
    # bt(i+1, pth+[nums[i]])   # 与以上三行等价
```



****

## [组合](https://leetcode.cn/problems/combinations/)

![image-20240214214102933](./assets/image-20240214214102933.png)

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(start, path):
            if len(path) == k:
                ans.append(path[:]) #别漏写[:]
                return
			#n+1可以优化n-(k-len(pth))+2
            for i in range(start, n+1): 
                path.append(i)
                backtrack(i+1, path) #是path，别写错
                path.pop()
        
        ans = []
        backtrack(1, []) # 是从1开始的
        return ans
```

> `ans.append(path[:])` 中使用 `path[:]` 原因：
>
> `path[:]` 会创建一个 `path` 的副本。如果直接使用 `ans.append(path)`，则 `ans` 列表中的每个元素都会指向同一个 `path` 列表对象，而不是其副本。这意味着在后续的迭代过程中，当我们改变 `path` 的内容时，`ans` 列表中的元素也会随之改变，这可能不是我们想要的行为。



## [组合总和 III](https://leetcode.cn/problems/combination-sum-iii/)

![image-20240212223356149](./assets/image-20240212223356149.png)

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        def bt(tot, start, pth):
            if tot>n: # 剪枝
                return
            
            if len(pth)==k and tot == n:
                ans.append(pth[:])
                return
            
            for i in range(start, 9-(k-len(pth))+2): # 剪枝
                pth.append(i)
                tot+=i
                bt(tot, i+1, pth)
                tot-=i    # 回溯
                pth.pop() # 回溯

        ans=[]
        bt(0,1,[])
        return ans
```



## [电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

![image-20240212220733336](./assets/image-20240212220733336.png)

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:return []

        mp = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }

        def bt(i,pth):
            if i == len(digits):
                ans.append("".join(pth[:]))
            else:
                d = digits[i]
                for s in mp[d]:
                    pth.append(s)
                    bt(i+1, pth)
                    pth.pop()
                    
        ans = []
        bt(0, [])
        return ans
```



## [组合总和](https://leetcode.cn/problems/combination-sum/) 

![image-20240212232001810](./assets/image-20240212232001810.png)

> **无重复元素** 的整数数组 `candidates`
>
> 2 <= candidates[i] <= 40

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def bt(target, start, pth):
            if target==0:
                ans.append(pth[:])
                return
            
            for i in range(start, len(candidates)):
                target -= candidates[i]
                if target < 0:
                    break
                pth.append(candidates[i])
                bt(target, i, pth) #重复使用元素，仍使用i
                target += candidates[i]
                pth.pop()

        candidates.sort()
        ans = []
        bt(target,0,[])
        return ans
```



## [组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:    	
        def bt(target, start, pth):
            if target == 0:
                ans.append(pth[:])
                return

            for i in range(start, len(candidates)):
                # 跳过同一树层使用过的元素
                if i>start and candidates[i] == candidates[i-1]:
                    continue
                target -= candidates[i]
                if target < 0:
                    break
                pth.append(candidates[i])
                bt(target, i+1, pth)
                target += candidates[i]
                pth.pop()

        candidates.sort()
        ans = []
        bt(target, 0, [])
        return ans
```



## [分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

![image-20240214223348814](./assets/image-20240214223348814.png)

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def bt(start, pth):
            if start == len(s):
                ans.append(pth[:])
                return
            
            for i in range(start, len(s)):
                if s[start: i+1] == s[start: i+1][::-1]:
                    pth.append(s[start:i+1])
                    bt(i+1, pth) 
                    pth.pop()  
        ans = []
        bt(0, [])
        return ans
```



## [复原 IP 地址](https://leetcode.cn/problems/restore-ip-addresses/)

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def bt(start, pth):
            if start == len(s) and len(pth) == 4:
                ans.append(".".join(pth))
                return

            for i in range(start, min(start+3, len(s))):
                if len(pth) > 4:  # 剪枝
                    break
                if self.is_valid(s, start, i):
                    pth.append(s[start:i+1])
                    bt(i+1, pth)
                    pth.pop()
        ans = []
        bt(0, [])
        return ans

    def is_valid(self, s, start, end):
        if s[start] == '0' and start != end:  # 0开头的数字不合法
            return False
        num = int(s[start:end+1])
        return 0 <= num <= 255
```

---

子集==没有return==，一下3题

## [子集](https://leetcode.cn/problems/subsets/) [==不==包含重复元素]

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def bt(start, pth):
            ans.append(pth[:])
            
            for i in range(start, len(nums)):
                pth.append(nums[i])
                bt(i+1, pth)
                pth.pop()
        
        ans = []
        bt(0, [])
        return ans
```



## [子集 II](https://leetcode.cn/problems/subsets-ii/) [==包含==重复元素]

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def bt(start, pth):
            ans.append(pth[:])

            for i in range(start, len(nums)):
                # 比上题多一个判断逻辑
                if i > start and nums[i] == nums[i-1]:
                    continue
                pth.append(nums[i])
                bt(i+1, pth)
                pth.pop()

        ans = []
        nums.sort() #排序
        bt(0, [])
        return ans
```



## [非递减子序列](https://leetcode.cn/problems/non-decreasing-subsequences/)

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        def bt(nums, pth):
            if len(pth) > 1:
                ans.append(pth[:])
                # 没有return
                
            tmp = set()
            for i, n in enumerate(nums):
                if n in tmp:
                    continue
                if not pth or n >= pth[-1]:
                    tmp.add(n)
                    bt(nums[i+1:], pth+[n])
        ans = []
        bt(nums, [])
        return ans
```

[思路2](https://leetcode.cn/problems/non-decreasing-subsequences/solutions/1389337/by-flix-cqav)

> 选不选 nums[i] 是去重策略

```python
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        def bt(i, pth):
            if i == len(nums):
                if len(pth) > 1:
                    ans.append(pth[:])     
                return
            
            # 【1】选 nums[i]
            if not pth or nums[i]>=pth[-1]: # 需满足递增
                pth.append(nums[i])         # 选nums[i]
                bt(i+1, pth)
                pth.pop()                   # 回溯复原
                # bt(i+1, pth+[nums[i]])   # 与以上三行等价
            
            # 【2】不选 nums[i]：
            # 只有在nums[i]不等于前一项tmp[-1]的情况下才考虑不选nums[i]
            # 即若nums[i] == pth[-1]，则必考虑选nums[i]，不予执行不选nums[i]的情况
            if not pth or (pth and nums[i] != pth[-1]): # 避免重复
                bt(i+1, pth)

        ans = []
        bt(0, [])
        return ans
```



## [全排列](https://leetcode.cn/problems/permutations/)

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:       
        def bt(nums, pth):
            if not nums:
                ans.append(pth[:])
                return 
            for i in range(len(nums)):
                bt(nums[:i] + nums[i+1:], pth + [nums[i]])
                
        ans = []
        bt(nums, [])
        return ans
```



## [全排列 II](https://leetcode.cn/problems/permutations-ii/)

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def bt(nums, pth):
            if not nums:
                ans.append(pth[:])
                return
            
            tmp = set()
            for i in range(len(nums)):
                if nums[i] in tmp:
                    continue
                bt(nums[:i]+nums[i+1:], pth+[nums[i]])
                tmp.add(nums[i])
        
        ans = []
        bt(nums, [])
        return ans
```



## [重新安排行程](https://leetcode.cn/problems/reconstruct-itinerary/)

```python
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        from collections import defaultdict
        mp = defaultdict(list)  
        
        for f, t in tickets:
            mp[f] += [t]         
        for f in mp:
            mp[f].sort()   
        
        def bt(f): 
            while mp[f]:
                bt(mp[f].pop(0))#路径检索
            ans.insert(0, f)    #放在最前
        
        ans = []
        bt('JFK') #题目必须从JFK开始
        return ans
```



## [N 皇后](https://leetcode.cn/problems/n-queens/)

> 如何判断是否在对角上呢?
>
> 正对角就是 (i,j) 相加之和一样的
>
> 负对角就是 (i,j) 相减只差一样的

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
                            #列，   正对角，  负对角
        def bt(i=0, pth=[], col=[], z=set(), f=set()):
            if i == n: #行
                ans.append(pth)
                return 
            
            for j in range(n):
                if j not in col \
                and i-j not in z \
                and i+j not in f:
                    bt(i+1, 
                    pth+[s[:j]+'Q'+s[j+1:]], 
                    col+[j], 
                    z|{i-j}, 
                    f|{i+j}) #并集
        ans = []
        s = '.' * n
        bt()
        return ans
```



## [解数独](https://leetcode.cn/problems/sudoku-solver/)

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def bt(i, j):
            """i, j代表遍历到的行、列索引"""
            if i == 9:   # 遍历完最后一行后，结束
                return True

            if j == 9:   # 遍历完最后一列后，转去遍历下一行
                return bt(i+1, 0)

            if board[i][j] != '.':  # 有数字
                return bt(i, j+1)

            for n in range(1, 10):  # 填空
                n = str(n)
                if not self.check(board, i, j, n):  
                    continue
                board[i][j] = n                 
                # 直接return是因为只需要一个可行解，而不需要所有可行解                
                if bt(i, j+1):    
                    return True
                board[i][j] = '.'  # 撤销选择
        bt(0, 0)

    def check(self, board, row, col, n):
        for i in range(9):
            if board[row][i] == n:
                return False
            if board[i][col] == n:
                return False
            r = (row//3)*3 + i // 3
            c = (col//3)*3 + i % 3
            if board[r][c] == n:
                return False
        return True
```



# 贪心算法

## [分发饼干](https://leetcode.cn/problems/assign-cookies/)

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        if len(s) == 0:
            return 0
            
        g.sort()
        s.sort()
       
        ans = 0

        for i in g[::-1]:
            if not s: 
                break
            if i <= s[-1]:
                s.pop()
                ans += 1
        return ans
```



## [摆动序列](https://leetcode.cn/problems/wiggle-subsequence/)

```python
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return len(nums)
        
        up, down = 1, 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                up = down + 1
            elif nums[i] < nums[i-1]:
                down = up + 1            
        return max(up, down)
```



## [最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]  
        current_sum = nums[0]  
        
        for n in nums[1:]:  # 从第二个元素开始遍历数组
            current_sum = max(n, current_sum + n)  
            max_sum = max(max_sum, current_sum) 
            
        return max_sum         
```



## [买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(1, len(prices)):
            tmp = prices[i] - prices[i - 1]
            if tmp > 0: 
                profit += tmp
        return profit
```



# 动态规划

==特别注意：==遍历的起始位置

## [斐波那契数](https://leetcode.cn/problems/fibonacci-number/)

```python
class Solution:
    def fib(self, n: int) -> int:
        if n == 0:  # 要加
            return 0
        
        dp = [0]*(n+1)
        dp[1] = 1
        
        for i in range(2,n+1): # 从2开始
            dp[i]=dp[i-1]+dp[i-2]
        return dp[n]
```



## [爬楼梯](https://leetcode.cn/problems/climbing-stairs/)

[通用解法](#plt)

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = [1]*(n+1)
        for i in range(2, n+1):
            dp[i]=dp[i-1]+dp[i-2]       
        return dp[-1]
```



## [使用最小花费爬楼梯](https://leetcode.cn/problems/min-cost-climbing-stairs/)

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0]*(len(cost)+1)

        for i in range(2, len(cost)+1):
            dp[i] = min(
                dp[i-1] + cost[i-1],
                dp[i-2] + cost[i-2])
        
        return dp[-1]
```



## [不同路径](https://leetcode.cn/problems/unique-paths/) 【无障碍】

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0]*(n) for i in range(m)]
        
        for i in range(m):
            dp[i][0]=1                           
        for j in range(n):
            dp[0][j]=1         
        
        for i in range(1,m): #注意开始为1
            for j in range(1,n): #注意开始为1
                dp[i][j] = dp[i-1][j]+dp[i][j-1]
        return dp[-1][-1]
```



## [不同路径 II](https://leetcode.cn/problems/unique-paths-ii/) 【有障碍】

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if obstacleGrid[0][0]==1 or obstacleGrid[-1][-1]==1: #
            return 0

        m,n = len(obstacleGrid),len(obstacleGrid[0])
        dp = [[0]*n for _ in range(m)]
        
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        for j in range(n):
            if obstacleGrid[0][j] == 1:
                break
            dp[0][j] = 1
        
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]
```



## [整数拆分](https://leetcode.cn/problems/integer-break/)

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0]*(n+1)
        dp[2]=1

        for i in range(3,n+1):
            for j in range(1, i//2+1): #从1开始
                dp[i] = max(
                    dp[i], 
                    (i-j) * j,     # i-j 不接着分割
                    dp[i-j] * j)   # i-j 接着分割 #只能怎么写
        return dp[-1]
```



## [不同的二叉搜索树](https://leetcode.cn/problems/unique-binary-search-trees/)

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp=[0]*(n+1)
        
        dp[0]=1
        dp[1]=1

        for i in range(2, n+1):
            for j in range(i):
                dp[i] += dp[j]*dp[i-j-1]
        
        return dp[n]
```



## 背包总结

01 --> 背包要==逆==序遍历。背包、物品先后无所谓

完全 --> 背包要==顺==序遍历先遍历物品是组合，反之是排序

<img src="./assets/20230310000726.png" alt="416.分割等和子集1"  />

![img](./assets/背包问题1.jpeg)

## 01背包

将一个集合分为两个部分

<img src="./assets/image-20240331160617427.png" alt="image-20240331160617427" style="zoom:67%;" />

> j - weight【i】，可以理解为背包需要留出这个物品 i 的容量才可以放物品 i

<img src="./assets/image-20240402144501849.png" alt="image-20240402144501849" style="zoom: 80%;" />

<img src="./assets/image-20240402145320918.png" alt="image-20240402145320918" style="zoom:80%;" />

```python
def test_2_wei_bag_problem1():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bagweight = 4

    # 二维数组
    dp = [[0] * (bagweight + 1) for _ in range(len(weight))]

    # 初始化
    for j in range(weight[0], bagweight + 1):
        dp[0][j] = value[0]

    # weight数组的大小就是物品个数
    for i in range(1, len(weight)):  # 遍历物品
        for j in range(bagweight + 1):  # 遍历背包容量
            if j < weight[i]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])

    print(dp[-1][-1])
```



```python
def test_1_wei_bag_problem():
    weight = [1, 3, 4]
    value = [15, 20, 30]
    bagWeight = 4

    # 初始化
    dp = [0] * (bagWeight + 1)
    for i in range(len(weight)):  # 遍历物品
        for j in range(bagWeight, weight[i]-1, -1):  # 遍历背包容量  # 倒叙遍历
            dp[j] = max(dp[j], dp[j-weight[i]] + value[i])

    print(dp[-1])
```



### [分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)

<img src="./assets/image-20240403111155882.png" alt="image-20240403111155882" style="zoom: 50%;" />

```python
def canPartition(self, nums: List[int]) -> bool:
    total = sum(nums)

    if total % 2 != 0:
        return False
    target = total//2
    
    dp = [False] * (target+1)
    dp[0] = True

    for num in nums:
        for i in range(target, num-1, -1):
            dp[i] = dp[i] or dp[i-num]
    return dp[target]
```



### [最后一块石头的重量 II](https://leetcode.cn/problems/last-stone-weight-ii/)

将这些数字分成两拨 (同上一题)，使得他们的和的差最小

```python
def lastStoneWeightII(self, stones: List[int]) -> int:
    tot = sum(stones)
    t = tot // 2

    dp = [0] * (t+1)
    for s in stones:
        for i in range(t, s-1, -1):
            dp[i] = max(
                dp[i],
                dp[i-s]+s
            )
    return tot - 2*dp[-1]
```



### <span id='mbh'>[目标和](https://leetcode.cn/problems/target-sum/)</span>

left = (target + sum)/2 

```python
def findTargetSumWays(self, nums: List[int], target: int) -> int:
    total_sum = sum(nums)  # 计算nums的总和
    if abs(target) > total_sum:
        return 0  # 此时没有方案
    if (target + total_sum) % 2 == 1:
        return 0  # 此时没有方案
    
    target_sum = (target + total_sum) // 2  
    dp = [0] * (target_sum + 1) 
    dp[0] = 1  # 当目标和为0时，只有一种方案，即什么都不选
    
    for num in nums:
        for j in range(target_sum, num-1, -1):
            dp[j] += dp[j - num]  # 状态转移方程，累加不同选择方式的数量
    return dp[target_sum]  # 返回达到目标和的方案数
```



### [一和零](https://leetcode.cn/problems/ones-and-zeroes/)[2个背包属性，01]

```python
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]  # 创建二维动态规划数组，初始化为0
        # 遍历物品
        for s in strs:
            ones = s.count('1')  # 统计字符串中1的个数
            zeros = s.count('0')  # 统计字符串中0的个数
            # 遍历背包容量且从后向前遍历
            for i in range(m, zeros-1, -1):
                for j in range(n, ones-1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)  # 状态转移方程
        return dp[m][n]
```



## 完全背包

### [零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/)(组合)

类似[目标和](#mbh)

> 补充：先遍历物品是组合，反之是排序
>
> 背包要正序遍历

```python
def change(self, amount: int, coins: List[int]) -> int:
    dp = [0] * (amount+1)
    dp[0]=1

    for coin in coins: # 先遍历物品
        for j in range(coin, amount+1):
            dp[j] += dp[j-coin]
    return dp[-1]
```



### [零钱兑换](https://leetcode.cn/problems/coin-change/)

```python
def coinChange(self, coins: List[int], amount: int) -> int:
    dp = [float('inf')] * (amount+1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount+1):
            dp[i] = min(
                dp[i],
                dp[i-coin]+1
            )
    return dp[-1] if dp[-1] != float('inf') else -1
```



### [完全平方数](https://leetcode.cn/problems/perfect-squares/)

```python
def numSquares(self, n: int) -> int:
    dp = [float('inf')] * (n + 1)
    dp[0] = 0

    for i in range(1, int(n**0.5)+1):  # 遍历物品
        for j in range(i*i, n+1):  # 遍历背包
            dp[j] = min(dp[j-i*i]+1, dp[j])
    return dp[-1]
```



### [单词拆分](https://leetcode.cn/problems/word-break/)

```python
def wordBreak(self, s: str, wordDict: List[str]) -> bool:       
    n=len(s)
    
    dp=[False]*(n+1)
    dp[0]=True
    
    for i in range(n): # 是n
        for j in range(i+1,n+1): # i+1，n+1
            if(dp[i] and (s[i:j] in wordDict)): # and
                dp[j]=True
    return dp[-1]
```



### [组合总和 Ⅳ](https://leetcode.cn/problems/combination-sum-iv/)（排序）

```python
def combinationSum4(self, nums: List[int], target: int) -> int:
    dp = [0] * (target + 1)  
    dp[0] = 1  

    for i in range(1, target+1): 
        for num in nums:  # 后遍历物品
            if i >= num:  
                dp[i] += dp[i-num]
    return dp[-1]
```



### <span id='plt'>爬楼梯（进阶版）【排序】</span>

**一步一个台阶，两个台阶，三个台阶，.......，直到 m个台阶。问有多少种不同的方法可以爬到楼顶呢？**

解法同上一题，也是==排序==

> 这其实是一个完全背包问题。
>
> 1阶，2阶，.... m阶就是==物品==，楼顶就是==背包==。

```python
def combinationSum4(self, m, target: int) -> int:
    dp = [0] * (target + 1)  
    dp[0] = 1  

    for i in range(1, target+1): 
        for j in range(1, m+1):  # 后遍历物品
            if i >= j:  
                dp[i] += dp[i-j]
    return dp[-1]
```



## 多重背包

![image-20240408151105522](./assets/image-20240408151105522.png)

```python
# bag_weight, n, weight, value, nums 
dp = [0] * (bag_weight + 1)

for i in range(n):
    for j in range(bag_weight, weight[i]-1, -1):
        for k in range(1, nums[i]+1):
            if j - k * weight[i] >= 0:
                dp[j] = max(
                    dp[j], 
                    dp[j - k*weight[i]] + k*value[i]
                    )
print(dp[bag_weight])
```



## [打家劫舍](https://leetcode.cn/problems/house-robber/)

dp[i]  =  max(    dp[i-1],    dp[i-2] + nums[i]]    )

```python
def rob(self, nums: List[int]) -> int:
    if len(nums) == 1:
        return nums[0]

    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(
        nums[0],
        nums[1]
    )

    for i, n in  enumerate(nums[2:]):
        dp[i+2] = max(
            dp[i+1],
            dp[i] + n
        )
    
    return dp[-1]
```

```python
def rob(self, nums: List[int]) -> int:
    cur, pre = 0, 0
    for num in nums:
        cur, pre = max(pre + num, cur), cur
    return cur
```

> nums = [2,23,9,3,20]，看成【0，0，2,23,9,3,20】



## [打家劫舍 II](https://leetcode.cn/problems/house-robber-ii/)

首位连成环：【1】不抢第一个；【2】不抢最后一个

```python
def rob(self, nums: [int]) -> int:
    def build(nums):
        cur, pre = 0, 0
        for num in nums:
            cur, pre = max(pre + num, cur), cur
        return cur
    return max(build(nums[:-1]), build(nums[1:])) if len(nums) != 1 else nums[0]
```



## [打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/)【树dp】

[后序遍历](#djjs)

![image-20240416132331982](./assets/image-20240416132331982.png)

>  dp为2维数组【0，1】【偷，不偷】
>
> 注意：_rob返回的是2维数组，返回的 `left = _rob(root.left)` 也是2维数组，故有 `left[1]`

```python
def rob(self, root: TreeNode) -> int:
    def _rob(root):
        if not root: 
            return 0, 0  # 偷，不偷
        left = _rob(root.left)
        right = _rob(root.right)

        v1 = root.val + left[1] + right[1] # 偷当前节点, 则左右子树都不能偷
        v2 = max(left) + max(right)        # 不偷当前节点, 则取左右子树中最大的值
        return v1, v2

    return max(_rob(root))
```

