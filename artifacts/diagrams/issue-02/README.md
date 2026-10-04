# T02 类图

`domain-objects.dot` 是可编辑 Graphviz 文件，`domain-objects.svg` 是导出图。空心三角指向父类；蓝色菱形表示引用；绿色虚线表示调用。R01–R24 对应 [Issue #2 证据](../../../docs/evidence/issues/issue-02.md)。

```bash
dot -Tsvg domain-objects.dot -o domain-objects.svg
```
