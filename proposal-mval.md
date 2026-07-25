# MValとデバイス論理長の設計提案

> [!NOTE]
> 実装時の最終決定では、長さとcapacityはユーザーが操作する値ではないため、公開APIから
> `len()`、`is_empty()`、`capacity()`をすべて除外した。論理extentと物理上限は内部でのみ保持・
> 伝播する。また、公開sliceは`DeviceSlice<R, T>` / `DeviceSliceMut<R, T>`とし、runtimeを型で
> 保持する。以下の`MIter::len()` / public `capacity()`に関する記述は、検討過程の旧案である。

## 概要

この提案では、`MVal<R, Item>` を「ホスト上で単値として解決でき、デバイス上では長さ1の
`MIter<R, Item = Item>` として読める値」と定義する。

`MVal` は値の物理的な所在を隠蔽する。

- ホスト値 `T` は、デバイス上では `lazy::constant(T).take(1)` として振る舞う
- `Scalar<R, T>` は、デバイス上の1要素storageをzero-copyで参照する
- encoded valueは、内部表現の長さ1 iteratorにlazyなdecode mapを重ねる
- `read()` は単値をホストへ返し、ホスト値なら転送せず、デバイス値なら必要なreadbackを行う

さらに、`MIter::len()` の戻り値を `MVal<R, MIndex>` にする。これにより、論理長がホスト上で
既知か、GPU上でのみ既知かにかかわらず、同じAPIで次の処理へ渡せる。

```text
as_iter()  値または論理長をGPUパイプライン内で使う
read()     値または論理長をホスト上で解決する
```

この設計の中心は、論理長と物理allocation上限を分離することである。

```text
len()       実際の論理長。host/device透過なMVal<MIndex>
capacity()  ホスト既知の物理上限。MIndex
```

GPU上の論理長だけを使って、新しいバッファを安全に確保したりdispatch上限を決めたりすることは
できない。そのため、`capacity()` は引き続きホスト既知の値として保持する。

## 目的

- ホスト値とデバイス値を同じアルゴリズム引数として扱う
- host valueをホストで読む場合に、不要なupload、kernel launch、readbackを発生させない
- device valueをホストへ戻さず、通常の`MIter`として次のGPUアルゴリズムへ渡す
- encoded valueの物理表現を、公開APIや後続アルゴリズムへ固定しない
- device-residentな論理長をreadbackせず、次のGPUアルゴリズムへ伝播させる
- length-changing algorithmが、論理長を得るためだけに同期することを避ける
- 単列だけでなく、flat row全体に同じ仕組みを適用する

## 非目的

- デバイス上の論理長だけから、上限不明のallocationを行うこと
- `Option<T>`など、storage能力を持たない`CubeType`を`MVec`へmaterializeすること
- encoded valueに共通の物理storage layoutを強制すること
- `MVal`をtrait objectとして型消去すること
- 任意の`MIter`を効率よくホスト上で評価すること

`MVal`は長さ1に限定された値の抽象である。一般の`MIter`をCPU interpreterとして評価する抽象ではない。

## MValの契約

提案する公開traitは次の形である。

```rust
pub trait MVal<R, Item>: Sized
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
{
    type Iter<'a>: MIter<R, Item = Item>
    where
        Self: 'a;

    /// デバイスアルゴリズムへ渡せる、論理長が必ず1のiteratorを返す。
    ///
    /// この操作自体はreadbackや同期を行わない。
    fn as_iter(&self) -> Self::Iter<'_>;

    /// この値をホスト上の単値として返す。
    ///
    /// ホスト値では転送を行わない。デバイス値では必要なreadbackと同期を行う。
    fn read(&self, exec: &Executor<R>) -> Result<Item, Error>;
}
```

`Item`をassociated typeではなくtrait parameterとして残す。これにより、
`impl MVal<R, T> for T`と`impl MVal<R, T> for Scalar<R, T>`を自然に共存させられる。

### 不変条件

すべての`MVal`実装は次を満たさなければならない。

1. `as_iter()`が返すiteratorの論理長は必ず1である
2. `as_iter()`のitemと`read()`の戻り値は、同じ論理値を表す
3. `as_iter()`はホストへのreadbackを行わない
4. `read()`は、値が既にホスト上にある場合はGPU処理を行わない
5. lazy decodeを持つ場合、device decodeとhost decodeは同じ意味を持つ

長さ1という不変条件を構造的に保証したい場合、内部でprivateな`One<Iter>` wrapperを使用できる。
`One`のconstructorを公開せず、`MVal`実装だけが生成する。

## ホスト値Tの実装

単一storage leafに対する概念的な実装は次のとおりである。

```rust
impl<R, T> MVal<R, T> for T
where
    R: Runtime,
    T: MStorageElement,
{
    type Iter<'a>
        = lazy::Taken<lazy::Constant<T>>
    where
        Self: 'a;

    fn as_iter(&self) -> Self::Iter<'_> {
        lazy::constant(*self).take(1)
    }

    fn read(&self, _exec: &Executor<R>) -> Result<T, Error> {
        Ok(*self)
    }
}
```

この実装では次の挙動になる。

```text
as_iter()  consumer kernelがstageするときだけ、1要素のconstant inputとしてuploadされる
read()     元のTをそのまま返し、GPU転送も同期も行わない
```

### flat rowへの一般化

実際の実装を`MStorageElement`だけに限定してはならない。tupleなどのflat rowも同じ能力を持つ必要がある。
そのため、ホスト上の1値を長さ1のread expressionへ変換する内部能力を分離する。

```rust
pub(crate) trait OneValueRead<R>: CubeType
where
    R: Runtime,
{
    type Iter<'a>: MIter<R, Item = Self>
    where
        Self: 'a;

    fn one_value_iter(&self) -> Self::Iter<'_>;
}
```

primitiveは単一のconstant leafになる。flat rowは値をstorage leavesへ分解し、各leafの
`constant(...).take(1)`をSoAとしてzipする。

```text
(u32, f32)
    ↓
zip2(
    constant(u32).take(1),
    constant(f32).take(1),
)
```

この能力は「任意長のowned storageを確保できる」という`MAlloc`の能力とは異なる。したがって、
ホスト値の`MVal`実装は`MAlloc`ではなく`OneValueRead`を要求する。

```rust
impl<R, T> MVal<R, T> for T
where
    R: Runtime,
    T: OneValueRead<R> + Clone,
{
    type Iter<'a> = T::Iter<'a> where Self: 'a;

    fn as_iter(&self) -> Self::Iter<'_> {
        self.one_value_iter()
    }

    fn read(&self, _exec: &Executor<R>) -> Result<T, Error> {
        Ok(self.clone())
    }
}
```

`as_iter(&self)`と`read(&self)`の両方が値を再利用するため、ホスト値には`Clone`相当の能力が必要になる。
現在対象となるphysical leavesとそのflat rowは実質的に`Copy`であるため、初期実装では`Copy`に
限定してもよい。

## Scalarの実装

`Scalar<R, T>`は、デバイス上で実際に`T`として読める1要素storageを所有する具体型とする。

```rust
pub struct Scalar<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
{
    storage: MVec<R, T>,
}
```

constructorではstorageの物理行数が1であることを保証する。

```rust
impl<R, T> MVal<R, T> for Scalar<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
{
    type Iter<'a>
        = <MVec<R, T> as MStorage<R>>::Slice<'a>
    where
        Self: 'a;

    fn as_iter(&self) -> Self::Iter<'_> {
        self.storage.slice(..)
    }

    fn read(&self, exec: &Executor<R>) -> Result<T, Error> {
        <T::Dispatch as ItemDispatch<R>>::read_value(exec, &self.storage)
    }
}
```

挙動は次のとおりである。

```text
as_iter()  既存のデバイスstorageをzero-copyで参照する
read()     1要素をreadbackし、ホスト上でTを返す
```

`MVal`のメソッドが`&self`を受け取るため、通常の利用では`Scalar`を消費しない。それでも
`impl MVal for &Scalar`を提供すると、`impl MVal`を値で受け取る既存APIへ明示的なborrowを渡せる。
これは`Scalar`の再利用を容易にするためのergonomicな実装として残す。

## decodeをlazy mapとして扱う

アルゴリズムが生成した内部表現と、公開上の論理値が異なる場合、decodeをstorage変換として
materializeしない。内部表現の長さ1 iteratorにlazy mapを重ねる。

例えば、検索結果を内部的に`MIndex::MAX`でencodeしている場合を考える。

```text
Scalar<MIndex>
    ↓ lazy map DecodeOptionalIndex
MIter<Option<MIndex>>
```

`Option<T>`は`CubeType`であるため、read-onlyな`MIter::Item`として使用できる。
`CubeElement`や`MAlloc`である必要はない。`Option<T>`を物理storageへ書く必要もない。

device decodeとhost decodeを対にする。

```rust
pub trait ValueMap<Input>: op::UnaryOp<Input> {
    fn apply_host(value: Input) -> Self::Output;
}
```

```rust
struct DecodeOptionalIndex;

#[cubecl::cube]
impl op::UnaryOp<MIndex> for DecodeOptionalIndex {
    type Output = Option<MIndex>;

    fn apply(value: MIndex) -> Option<MIndex> {
        (value != MIndex::MAX).then_some(value)
    }
}

impl ValueMap<MIndex> for DecodeOptionalIndex {
    fn apply_host(value: MIndex) -> Option<MIndex> {
        (value != MIndex::MAX).then_some(value)
    }
}
```

値のmap adapterを用意する。

```rust
pub(crate) struct MappedValue<Value, Op> {
    value: Value,
    _op: PhantomData<Op>,
}
```

```rust
impl<R, Input, Output, Value, Op> MVal<R, Output>
    for MappedValue<Value, Op>
where
    R: Runtime,
    Input: CubeType + Send + Sync + 'static,
    Output: CubeType + Send + Sync + 'static,
    Value: MVal<R, Input>,
    Op: ValueMap<Input, Output = Output>,
    for<'a> lazy::Map<Value::Iter<'a>, Op>: MIter<R, Item = Output>,
{
    type Iter<'a>
        = lazy::Map<Value::Iter<'a>, Op>
    where
        Self: 'a;

    fn as_iter(&self) -> Self::Iter<'_> {
        lazy::map(self.value.as_iter(), Op)
    }

    fn read(&self, exec: &Executor<R>) -> Result<Output, Error> {
        Ok(Op::apply_host(self.value.read(exec)?))
    }
}
```

device consumerではdecode mapがconsumer kernelへ融合される。ホストで`read()`した場合は、内部表現だけを
readbackして`apply_host`を実行する。このため、decode結果をGPU上へmaterializeする追加kernelは不要である。

この設計では、sentinelなどの内部encodingはproducerごとに自由に選べる。共通のoptional storage layoutを
全アルゴリズムへ強制しない。

### device decodeとhost decodeの一致

`UnaryOp::apply`と`apply_host`は別実装になるため、意味の不一致を防ぐテストが必要である。

- 代表値と境界値についてdevice decodeとhost decodeを比較する
- sentinel、overflow、NaNなど、encoding固有の境界を明示的にテストする
- 可能ならdecode仕様を小さな共通primitiveから構成する

## アルゴリズムの引数

reduce、exclusive scan、fillなどの単値引数は、`Scalar`への先行変換を行わず、`MVal::as_iter()`を
直接利用する。

```rust
pub fn reduce<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    init: impl MVal<R, Input::Item>,
    op: Op,
) -> Result<impl MVal<R, Input::Item>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Op: ReductionOp<Input::Item>,
{
    let init_read = init.as_iter();
    // init_readを長さ1の入力としてreduce implementationへ渡す
    // ...
}
```

アルゴリズム内部で本当にownedな1要素storageが必要な場合だけmaterializeする。`MVal`の入口で
無条件に`into_device()`する設計にはしない。

## アルゴリズムの戻り値を隠蔽する

公開APIは、具体的な`Scalar`、内部encoding、`MappedValue`を戻り値型として公開しない。

```rust
pub fn find_if<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    pred: Pred,
) -> Result<impl MVal<R, Option<MIndex>>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: PredicateOp<Input::Item>,
{
    let encoded: Scalar<R, MIndex> = find_if_encoded(exec, input, pred)?;
    Ok(MappedValue::<_, DecodeOptionalIndex>::new(encoded))
}
```

利用者に見える契約は`MVal<R, Option<MIndex>>`だけである。

```rust
let found = find_if(&exec, input, pred)?;

let device_value = found.as_iter();
let host_value: Option<MIndex> = found.read(&exec)?;
```

return-position `impl Trait`のhidden concrete typeは関数ごとに一つでなければならない。同一関数の
分岐から`T`と`Scalar<T>`を直接返すことはできない。分岐が必要な場合は、次のいずれかを行う。

- private enumで共通化する
- 常に同じ内部表現へ正規化する
- 共通のadapterを分岐の外側に置く

GPUアルゴリズムの結果は、通常は常に`MappedValue<Scalar<D>, Decode>`へ揃えられる。

## MIter::len()の変更

`MIter`は論理長の表現型をGATとして持つ。

```rust
pub trait MIter<R: Runtime>: Clone + Sized {
    type Item: CubeType + Send + Sync + 'static;

    type Len<'a>: MVal<R, MIndex>
    where
        Self: 'a;

    /// 実際の論理行数を返す。
    ///
    /// 戻り値のread()は必要に応じて同期するが、len()自体は同期しない。
    fn len(&self) -> Result<Self::Len<'_>, Error>;

    /// allocationとdispatchに使用できる、ホスト既知の物理上限。
    fn capacity(&self) -> Result<MIndex, Error>;

    // Item、Read、Slice、lower_readなどは従来どおり
}
```

`Result`は、zipの構造的不一致、長さoverflowなどの検査を残すために維持する。
device-residentであること自体はエラーではないため、`Error::UnresolvedLength`は不要になる。

### host-known length

固定長のiteratorは`MIndex`をそのまま返す。

```rust
impl<R, T> MIter<R> for FixedIterator<T>
where
    R: Runtime,
{
    type Len<'a> = MIndex where Self: 'a;

    fn len(&self) -> Result<MIndex, Error> {
        Ok(self.len)
    }

    fn capacity(&self) -> Result<MIndex, Error> {
        Ok(self.len)
    }
}
```

```rust
let len = input.len()?;

let len_input = len.as_iter(); // constant(len).take(1)
let host_len = len.read(&exec)?; // 即座にMIndexを返す
```

### device-known length

GPUが計算した論理長を保持するiteratorは、`Scalar<R, MIndex>`またはそれをborrowしたMValを返す。

```rust
impl<R, T> MIter<R> for DynamicIterator<R, T>
where
    R: Runtime,
{
    type Len<'a> = &'a Scalar<R, MIndex> where Self: 'a;

    fn len(&self) -> Result<Self::Len<'_>, Error> {
        Ok(&self.logical_len)
    }

    fn capacity(&self) -> Result<MIndex, Error> {
        Ok(self.capacity)
    }
}
```

```rust
let len = input.len()?;

let len_input = len.as_iter(); // 既存のdevice scalarを参照
let host_len = len.read(&exec)?; // ここで初めてreadback
```

### iterator adapterにおける長さの伝播

長さを変えないadapterは入力の`Len`をそのまま伝播する。

```text
Map<Input>      Len = Input::Len
Reverse<Input>  Len = Input::Len
```

出力行数を別のiteratorが決めるadapterは、そのiteratorの`Len`を伝播する。

```text
Permute<Values, Indices>  Len = Indices::Len
Gather                    Len = Indices::Len
```

sliceは元の長さにhost-knownな`start`と`limit`を適用したderived MValを返す。

```text
slice_len = min(source_len.saturating_sub(start), limit)
```

source lengthがhost valueなら、この計算はホスト上で即座に解決する。source lengthがdevice valueなら、
`zip`とlazy mapで長さ1のdevice expressionとして表現する。

```text
zip3(
    source_len.as_iter(),
    constant(start).take(1),
    constant(limit).take(1),
).map(ClampSliceLength)
```

このdecodeはconsumer kernelへ融合できる。ホストで`read()`する場合は、source lengthだけをreadbackして
同じ計算をホスト上で行う。

## host/deviceの選択とRustの具体型

あるiterator型が常にhost-known lengthを持つなら、`Len = MIndex`を選べる。常にdevice-knownなら、
`Len = Scalar<R, MIndex>`を選べる。

一方、現在の`DeviceSlice<T>`のように、同じ具体型の内部状態としてhost/device両方のlogical extentを
持ち得る場合、`len()`の具体的な戻り値型を実行時に変えることはできない。

```rust
// 同一implの戻り値としては不可能
if host_known {
    MIndex
} else {
    Scalar<R, MIndex>
}
```

associated typeおよび`impl Trait`のhidden typeは、implごとに一つの具体型へ固定されるためである。

この場合はprivateなsum typeを使用する。

```rust
pub(crate) enum LogicalLen<'a, R: Runtime> {
    Host(MIndex),
    Device(&'a Scalar<R, MIndex>),
}
```

`LogicalLen`自身が`MVal<R, MIndex>`を実装する。

```text
LogicalLen::Host
  as_iter() → constant(len).take(1)
  read()    → lenを即座に返す

LogicalLen::Device
  as_iter() → Scalar::as_iter()
  read()    → Scalar::read()
```

`as_iter()`のassociated typeを統一するため、privateな`LogicalLenIter`を使用できる。host constantと
device sliceはいずれも1つの`MIndex` read slotへloweringできるため、共通の1-slot read ABIとして
実装可能である。

このprivate型は公開シグネチャから隠す。

```rust
fn len(&self) -> Result<impl MVal<R, MIndex> + '_, Error>
```

または、`MIter::Len<'a>`としてのみ露出させる。

型レベルで必ず`MIndex`または`Scalar`を選択したい場合は、iteratorまたはvectorの型自体を
host-length版とdevice-length版に分ける必要がある。ただし、全adapterへ長さ表現の型が伝播し、
型とmonomorphizationが複雑になるため、初期実装ではprivate sum typeを許容する。

## capacityと論理長

`len()`をMValにしても、`capacity()`はホスト既知でなければならない。

```text
0 <= len <= capacity
```

これをすべてのMIterとstorageの不変条件とする。

例えばselection結果では次のようになる。

```text
capacity  入力の論理上限N
len       GPUが計算したselected_count
storage   N行分のupper-bound allocation
```

次のアルゴリズムは`capacity`でdispatchし、kernel内では`len.as_iter()`から得た値で有効範囲を制限する。

```rust
let capacity = input.capacity()?;
let len = input.len()?;

launch_with_upper_bound(
    exec,
    capacity,
    len.as_iter(),
    input,
    output,
)?;
```

この形なら、selection countをホストへ戻さずにmap、scan、sortなどの後続処理へ進める。

## owned vectorの意味

現在の`MStorage`は、owned storageについて物理allocation長と論理長が常に一致することを前提としている。
length-changing algorithmがdevice-resident lengthを保持したまま公開結果を返すには、この前提を
見直す必要がある。

提案後は、owned vectorも次の二つを持ち得る。

```text
capacity  実際に確保された行数
len       初期化済みの論理prefix長
```

固定長の通常vectorでは`len = capacity = MIndex`である。動的長の結果では、
`len`だけがdevice-residentなMValになる。

公開owned vectorを常にexact allocationのまま維持すると、device lengthをホストへreadbackして
compactする必要があり、この提案の効果が内部pipelineに限定される。そのため`MVec`自体が
host-knownな`capacity`とMValな論理`len`を持つ設計を採用する。別の`BoundedMVec`型は導入せず、
すべてのアルゴリズムで同じ`MIter`として扱う。

### ホスト転送

動的長vectorをホストへ転送する場合は、いずれにせよ論理長のreadbackが必要である。

```text
1. len.read(exec)
2. 有効prefixだけをホストへ転送
```

これはホスト値を要求する明示的な同期境界なので問題ない。GPUパイプライン内ではこの同期を行わない。

### 物理allocation

公開vectorは論理長を保持するため、物理allocationを論理長に合わせてcompactする公開操作は提供しない。
内部表現上exactなallocationが必要な箇所に限り、crate-privateな処理としてdevice lengthのreadbackと
prefix copyを行う。

## zipと長さの一致

二つのdevice-resident lengthが等しいかどうかを、readbackなしでホストの`Result`として返すことは
できない。zipの長さ検査では、次の区別が必要である。

- 両方host-knownなら、その場で比較する
- 同じdevice scalarまたは同じderived lengthを共有しているなら、provenanceから同一と判断する
- 一方がhost-knownなcapacityで、もう一方のdevice upper boundを安全に覆うだけでは、論理長の
  等値までは保証しない
- 無関係な二つのdevice lengthは、同期なしには等値を証明できない

初期実装では、現在の`LogicalExtent::zipped`と同様にprovenanceで同値を証明できる場合だけ受理する。
無関係なdevice lengthを暗黙に`min`として扱ってはならない。必要なら、意味の異なる明示的な
`zip_min`操作として設計する。

## is_emptyなどの派生値

`len()`がMValを返す場合、readbackしない`is_empty()`も単なるboolを返せない。

```text
is_empty = len.map(IsZero)
```

したがって、次のいずれかにする。

```rust
fn is_empty(&self) -> Result<impl MVal<R, bool> + '_, Error>;
```

または、`len()`から利用者が明示的にmapする。

```rust
let empty = input.len()?.map(IsZero);
```

ホスト上のboolが必要な場合だけ`empty.read(exec)`を呼ぶ。

## CubeCL上の実行モデル

この提案では、host valueとdevice valueのどちらもconsumer kernelから見ると長さ1のread inputになる。

```text
host T
  → stage時に1要素のconstant bindingを作る

Scalar<T>
  → 既存の1要素buffer handleをbindingする

MappedValue<D, Decode>
  → Dのbindingを読み、consumer kernel内でDecodeを適用する
```

hostとdeviceの違いはstagingに閉じる。consumer algorithmは`MIter<Item = T>`としてのみ扱う。

decode結果が`Option<T>`のようにstorage能力を持たなくても、consumer kernel内のread-onlyな
`CubeType`として利用できる。write boundaryを越えるアルゴリズムだけが、出力itemに`MAlloc`などの
storage能力を要求する。

長さのhost/device切り替えも同じ1-slot ABIで実装できる。これにより、長さの所在ごとに
アルゴリズム本体を複製する必要はない。

## コンパイル時間への配慮

長さ表現をすべて型レベルで伝播すると、iterator adapterの型がさらに大きくなり、
monomorphizationが増える可能性がある。

次の方針を採用する。

- fixed lengthは`MIndex`として単純化する
- runtimeにhost/device両方を持ち得る型はprivate sum typeへまとめる
- device lengthのread ABIは1-slotへ正規化する
- 長さ演算のためにアルゴリズム本体をhost/device別に複製しない
- `impl MVal`でproducer固有のdecode型を公開APIから隠す
- 同じ演算を繰り返しmaterializeする必要がある場合だけ`Scalar`へ正規化する

型レベルの純粋さより、公開APIの単純さとコンパイル時間を優先する。

## API利用例

### reduceの結果をGPU上で再利用する

```rust
let sum = vector::reduce(&exec, input, 0_u32, Add)?;

let repeated = lazy::permute(
    sum.as_iter(),
    lazy::constant(0).take(4),
);

let output = vector::map(&exec, repeated, op::Identity)?;
```

`sum.read(&exec)`を呼ばない限り、readbackは発生しない。

### ホスト値をMValとして渡す

```rust
let output = vector::exclusive_scan(
    &exec,
    input,
    0_u32,
    Add,
)?;
```

`0_u32.as_iter()`はconsumer kernelのconstant inputになる。`0_u32.read(&exec)`相当の操作は値を
そのまま返す。

### device-resident lengthを次のkernelへ渡す

```rust
let selected = vector::copy_where(&exec, input, pred)?;
let len = selected.len()?;

next_algorithm(
    &exec,
    selected,
    len.as_iter(),
)?;
```

`next_algorithm`は`selected.capacity()`をdispatch上限として使用し、`len.as_iter()`を有効prefix長として
使用する。

### 論理長をホスト上で取得する

```rust
let len = selected.len()?.read(&exec)?;
```

固定長なら即座に返り、device-resident lengthならここで初めて同期する。

## 移行手順

1. `MVal`を`as_iter()`と`read()`を持つtraitへ変更する
2. host value用の`OneValueRead`を単一leafとflat rowへ実装する
3. directな`Scalar<R, T>`を長さ1の`MIter` viewとして実装する
4. encoded resultを`MappedValue`と`ValueMap`へ移す
5. public algorithmの戻り値を必要に応じて`impl MVal<R, T>`へ変更する
6. 単値引数の先行`into_device()`を除去し、`as_iter()`を直接消費する
7. `MIter`へ`Len<'a>`を追加し、固定長iteratorから移行する
8. 現在の`LogicalExtent`をMValとして表現し、device lengthを`Scalar`またはprivate sum typeへ接続する
9. length-changing algorithmから不要なexact-prefix readbackを除去する
10. `capacity`と論理`len`を前提に、全kernelのdispatchと境界条件を検証する
11. `Error::UnresolvedLength`と旧`into_device()`経路を削除する

移行中は、旧APIと新APIを同時に維持するcompatibility layerを最小限にする。二重の値抽象を長期間
残さない。

## テスト方針

### MVal

- host valueの`read()`が値をそのまま返す
- host valueの`as_iter()`が長さ1である
- `Scalar::as_iter()`が長さ1である
- `Scalar::read()`が正しい値を返す
- borrowed `Scalar`を複数アルゴリズムで再利用できる
- flat rowのhost valueと`Scalar`が同じ結果になる
- `MappedValue::as_iter()`のdevice decodeと`read()`のhost decodeが一致する
- `Option<MIndex>`の`Some`、`None`、境界値を検証する

### 論理長

- fixed lengthの`read()`が同期を必要とせず正しい値を返す
- device lengthの`as_iter()`を後続kernelへ渡せる
- device lengthの`read()`が正しい値を返す
- 常に`len <= capacity`である
- empty、full、部分選択の各ケースを検証する
- dynamic lengthを持つ単列・多列vectorで同じ結果になる
- slice後のdevice lengthがsaturating semanticsに従う
- provenanceが同じdynamic lengthのzipを受理する
- 無関係なdynamic lengthを誤って同一または`min`として扱わない
- host転送が論理prefixだけを返す
- 明示的なexact化が正しいallocation長を返す

### 性能と同期

- host valueの`read()`がGPU commandを発行しない
- host-known lengthの`read()`がGPU commandを発行しない
- device lengthを次のalgorithmへ渡すだけではreadbackしない
- encoded valueのdevice decodeがconsumer kernelへ融合される
- length-changing pipelineの中間で不要なhost synchronizationが発生しない

## 未決事項

- public owned vectorを`capacity != len`まで一般化するか、`BoundedMVec`を分けるか
- `MVal::Iter`の長さ1不変条件をdocumentationだけで保証するか、privateな`One<Iter>`で保証するか
- `OneValueRead`を公開能力にするか、private implementation detailにするか
- `ValueMap::apply`と`apply_host`の一致をどの程度共通コード化できるか
- `is_empty()`などの派生値をMValとして直接提供するか
- unrelatedなdevice length同士のzipに明示的なvalidation APIを用意するか
- `MIter::len()`を`Result`のまま維持するか、不変条件の構築時検証によってinfallibleにするか

## 結論

`MVal`を「長さ1のMIter viewと、単値を得るread操作」として定義すると、ホスト値、device scalar、
encoded scalarを一つのモデルで扱える。

同じモデルを`MIter::len()`へ適用すると、論理長の所在をAPIから隠しながら、GPU上の長さを
readbackせず後続kernelへ渡せる。

設計上の要点は次の三つである。

1. `as_iter()`はGPU pipeline、`read()`は明示的なhost boundaryとする
2. `len()`と`capacity()`を分離し、device lengthでも安全な上限を必ず保持する
3. concreteな`Scalar`、decode、host/device sum typeは`impl MVal`の背後へ隠す

これにより、値と論理長の両方について、不要な同期を避けながらhost/device透過なAPIを構築できる。
