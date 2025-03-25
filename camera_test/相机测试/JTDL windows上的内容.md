- [ ] JTDL的硬件情况： 加内存， 先加两条8G的内存， 加到 16GB内存。

# 感觉用Windows 11  还是很舒服的。

- [ ] 稳定，不会挂掉。

- [ ] 效率高

- [ ] 睡眠一秒唤醒 

- [ ] 操作好、界面美观。

- [ ] 兼容性好。

### Windows的数据迁移，两台Windows电脑， 笔记本的数据迁移到 JTDL上？

怎么把一台windows电脑上的数据迁移同步？

- [ ] 有没有必要？
  
  - [ ] 如果有必要，要怎么迁移？
    
    - [ ] 不同的方法好像都很麻烦，好用的方法似乎要收费。[怎么将已安装的电脑软件迁移到另一台电脑？4种方法！](https://www.disktool.cn/content-center/transfer-installed-program-to-another-pc-windows-10-666.html)    [移动到 Windows 10 电脑 - Microsoft 支持](https://support.microsoft.com/zh-cn/windows/%E7%A7%BB%E5%8A%A8%E5%88%B0-windows-10-%E7%94%B5%E8%84%91-294fb3cb-7f2d-fd5a-5683-76aa499c8459)  

### win11使用

很多开发工具：

[开发人员主页计算机配置 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/dev-home/setup)

[面向 Windows 开发人员的开发人员主页 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/dev-home/#dev-home-dashboard-widgets)

### 修复这个很多登陆的问题

参照这个输入命令就可以了：[Windows11 0x80190001错误解决_0x80190001解决方案 win11-CSDN博客](https://blog.csdn.net/qq_36393978/article/details/124248158)

### 电脑的系统规划：

- [ ] MBP 笔记本： 
  
  - [ ] MAC
  
  - [ ] Windows11

- [ ] 

- [ ] JTDL 台式机
  
  - [ ] Linux
  
  - [ ] Windows10

- [ ] JT iMAC  台式机
  
  - [ ] MAC
  
  - [ ] Windows10

- [ ] 用windows双系统，不在Linux里用虚拟机
  
  - [ ] 升级到 windows 11 ， 参考这个得知对 硬件的要求：   [Windows 11 规格和系统需求| Microsoft](https://www.microsoft.com/zh-tw/windows/windows-11-specifications)
    
    - [ ] BIOS里开启一个TPM选项即可，主板手册里搜索TPM，有一个选项的。 [在电脑上启用 TPM 2.0 - Microsoft 支持](https://support.microsoft.com/zh-cn/windows/%E5%9C%A8%E7%94%B5%E8%84%91%E4%B8%8A%E5%90%AF%E7%94%A8-tpm-2-0-1fd5a332-360d-4f46-a1e7-ae6b0c90645c)
    
    - [ ] 为什么？
      
      - [ ] 更好的交互
      
      - [ ] 对WSL的支持
      
      - [ ] 对虚拟化更好的支持
      
      - [ ] Power Shell 等 也很好。
      
      - [ ] 节约时间。

### 修复JTDL 电脑的windows入口

- [x] 这个太复杂了，看了也浪费时间： [多系统启动管理：rEFInd-次世代BUG池](https://neucrack.com/p/63)；  [实体机安装双系统多系统教程 及引导修复指南-次世代BUG池](https://neucrack.com/p/330)  ；  [修复双启动选项电脑上的启动菜单 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows-hardware/manufacture/desktop/repair-the-boot-menu-on-a-dual-boot-pc?view=windows-11)

- [ ] 通过问GPT得知： 
  
  - [ ] 自己的jtdl电脑里的 时 Legacy 和 UEFI格式混合的，所以导致 Easy BCD 和 Grub都找不到对应的启动项，手动添加也不行的。
    
    - [ ] Linux时 Legacy模式按照的，硬盘时MBR格式；
      
      - [ ] 外挂的 三星硬盘居然也是GPT格式的硬盘。
    
    - [ ] Windows是 UEFI格式安装的， GPT格式的硬盘。

- [ ] 解决方案： 在Bios界面启动时选择， 在进入MSI的 Bios启动界面前，  按 F11按键，可以选择启动设备，然后选择第二个选项 就是Windows的入口了！！！   其他重装Windows或者 Linux都太折腾了。

### 之前想的电脑设备和资料文件 的安排和规划

- [ ] 还没想好： 30min 决定好！！！！
  
  - [ ] 参考这个： [公司、家里、笔记本资料同步方案整理 - EasonJim - 博客园](https://www.cnblogs.com/EasonJim/p/7802662.html)   ；  
  
  - [ ] 还有这个，非常好，一个人走了5年的弯路的经验： [2023 年电脑设备配置分享 — huangz.blog](https://huangz.blog/2023/working-set-up.html)

### 之前看的网络安全问题

大佬的思路： [如何对付公司的监控\[2\]：规避“主机行为审计” @ 编程随想的博客](https://program-think.blogspot.com/2013/05/howto-anti-it-audit-2.html)









MAC台式机 +  





Linux： 必须物理机，要用GPU+图形界面。

MAC： 必须物理机，虚拟机 黑不了、不敢用。



Windows： 可以在虚拟机里运行，但是如果在虚拟机里也不能用GPU打游戏。



JTDL： Linux物理机 + Windows虚拟机

MBP： MAC物理机 +  Windows物理机。

IMAC： MAC物理机 +  局域网 远程桌面到 Window、远程桌面到Linux。
