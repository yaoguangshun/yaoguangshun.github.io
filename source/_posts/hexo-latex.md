---
title: hexo配置LaTeX公式
date: 2018-5-15 21:12:18
tags: [LaTeX, hexo]
mathjax: true
categories: hexo
---

&emsp;&emsp;正常的hexo框架在默认情况下渲染数学公式会有很多问题，可以通过将hexo默认的引擎 `hexo-renderer-marked`更换为`hexo-renderer-kramed`来渲染markdown。
&emsp;&emsp;首先要将之前的`hexo-renderer-marked`卸载，并安装`hexo-renderer-kramed`。
```
npm uninstall hexo-renderer-marked --save
npm install hexo-renderer-kramed --save
```
&emsp;&emsp;在这之后建议在hexo的根目录下找到`package.json`文件，用文本编辑器打开它，删除字符串`hexo-renderer-marked`所在的一行并保存。之所以不直接卸载`hexo-renderer-marked`，是因为其他重要包极有可能在卸载该包的同时被删除。 
&emsp;&emsp;不要忘了行内公式的转义字符，打开`./node_modules/kramed/lib/rules`，并修改`inline.js`文件的11和20行，分别修改为
```
escape: /^\\([`*\[\]()#$+\-.!_>])/,
em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
```
&emsp;&emsp;每次在写文章前，要在YAML font-matter中添加`mathjax: true`，这样便能确保启动mathjax引擎进行渲染了。

