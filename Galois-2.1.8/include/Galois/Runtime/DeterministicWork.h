/** Deterministic execution -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */
#ifndef GALOIS_RUNTIME_DETERMINISTICWORK_H
#define GALOIS_RUNTIME_DETERMINISTICWORK_H

#include "Galois/Threads.h"

#include "Galois/ParallelSTL/ParallelSTL.h"
#include "Galois/TwoLevelIterator.h"
#include "Galois/Runtime/ContextPool.h"
#include "Galois/Runtime/ll/gio.h"

#include <boost/utility/enable_if.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <deque>
#include <queue>

namespace GaloisRuntime {
//! Implementation of deterministic execution
namespace DeterministicImpl {

struct OrderedTag { };
struct UnorderedTag { };

template<typename T>
struct DItem {
  T val;
  unsigned long id;
  void *localState;

  DItem(const T& _val, unsigned long _id): val(_val), id(_id), localState(NULL) { }
};

template<typename T, typename CompareTy>
struct DeterministicContext: public GaloisRuntime::SimpleRuntimeContext {
  typedef DItem<T> Item;

  Item item;
  const CompareTy* comp;
  bool not_ready;

  DeterministicContext(const Item& _item, const CompareTy& _comp): 
      SimpleRuntimeContext(true), 
      item(_item),
      comp(&_comp),
      not_ready(false)
  { }

  bool notReady() const { 
    return not_ready;
  }

  virtual void sub_acquire(GaloisRuntime::Lockable* L) {
    // TODO: should be redundant with new context hierarchy and virtual function call
    // Normal path
    // if (pendingFlag.flag.data == NON_DET) {
      // GaloisRuntime::SimpleRuntimeContext::acquire(L);
      // return;
    // }

    // Ordered and deterministic path
    if (getPending() == COMMITTING)
      return;

    if (L->Owner.try_lock()) {
      assert(!L->next);
      L->next = locks;
      locks = L;
    }

    DeterministicContext* other;
    do {
      other = static_cast<DeterministicContext*>(L->Owner.getValue());
      if (other == this)
        return;
      if (other) {

        bool conflict = (*comp)(*other, *this); // *other < *this
        if (conflict) {
          // A lock that I want but can't get
          not_ready = true;
          return; 
        }
      }
    } while (!L->Owner.stealing_CAS(other, this));

    // Disable loser
    if (other)
      other->not_ready = true; // Only need atomic write

    return;
  }
};

template<typename T, typename CompTy> 
struct OrderedContextComp {
  typedef DeterministicContext<T, OrderedContextComp> DetContext;

  const CompTy& comp;

  explicit OrderedContextComp(const CompTy& c): comp(c) {}

  inline bool operator()(const DetContext& left, const DetContext& right) const {
    return comp(left.item.val, right.item.val);
  }
};

template<typename T>
struct UnorderedContextComp {
  typedef DeterministicContext<T, UnorderedContextComp> DetContext;

  inline bool operator()(const DetContext& left, const DetContext& right) const {
    return left.item.id < right.item.id;
  }
};

template<typename Function1Ty,typename Function2Ty>
struct Options {
  static const bool needsStats = ForEachTraits<Function1Ty>::NeedsStats || ForEachTraits<Function2Ty>::NeedsStats;
  static const bool needsPush = ForEachTraits<Function1Ty>::NeedsPush || ForEachTraits<Function2Ty>::NeedsPush;
  static const bool needsBreak = ForEachTraits<Function1Ty>::NeedsBreak || ForEachTraits<Function2Ty>::NeedsBreak;
};

template<typename _T,typename _Function1Ty,typename _Function2Ty,typename _CompareTy>
struct OrderedOptions: public Options<_Function1Ty,_Function2Ty> {
  typedef _Function1Ty Function1Ty;
  typedef _Function2Ty Function2Ty;
  typedef _T T;
  typedef _CompareTy CompareTy;
  typedef OrderedContextComp<T, CompareTy> ContextComp;
  typedef DeterministicContext<T, ContextComp> DetContext;
  static const bool useOrdered = true;
  typedef OrderedTag Tag;

  Function1Ty fn1;
  Function2Ty fn2;
  CompareTy comp;
  ContextComp contextComp;

  OrderedOptions(const Function1Ty& fn1, const Function2Ty& fn2, const CompareTy& comp):
    fn1(fn1), fn2(fn2), comp(comp), contextComp(comp) { }

  DetContext make_context(const DItem<T>& item) const {
    return DetContext(item, contextComp);
  }
};

template<typename _T,typename _Function1Ty,typename _Function2Ty>
struct UnorderedOptions: public Options<_Function1Ty,_Function2Ty> {
  typedef _Function1Ty Function1Ty;
  typedef _Function2Ty Function2Ty;
  typedef _T T;
  typedef UnorderedContextComp<T> ContextComp;
  typedef DeterministicContext<T, ContextComp> DetContext;
  static const bool useOrdered = false;
  typedef UnorderedTag Tag;
  
  struct DummyCompareTy {
    bool operator()(const T&, const T&) const {
      return false;
    }
  };

  typedef DummyCompareTy  CompareTy;

  Function1Ty fn1;
  Function2Ty fn2;
  CompareTy comp;
  ContextComp contextComp;

  UnorderedOptions(const Function1Ty& fn1, const Function2Ty& fn2): fn1(fn1), fn2(fn2) { }

  DetContext make_context(const DItem<T>& item) const {
    return DetContext(item, contextComp);
  }
};

//! Some template meta programming
template<typename T>
struct has_local_state {
  typedef char yes[1];
  typedef char no[2];
  template<typename C> static yes& test(typename C::LocalState*);
  template<typename> static no& test(...);
  static const bool value = sizeof(test<T>(0)) == sizeof(yes);
};

template<typename T,typename FunctionTy,typename Enable=void>
struct StateManager { 
  void alloc(GaloisRuntime::UserContextAccess<T>&, FunctionTy& self) { }
  void dealloc(GaloisRuntime::UserContextAccess<T>&) { }
  void save(GaloisRuntime::UserContextAccess<T>&, void*&) { }
  void restore(GaloisRuntime::UserContextAccess<T>&, void*) { } 
};

template<typename T,typename FunctionTy>
struct StateManager<T,FunctionTy,typename boost::enable_if<has_local_state<FunctionTy> >::type> {
  typedef typename FunctionTy::LocalState LocalState;
  void alloc(GaloisRuntime::UserContextAccess<T>& c,FunctionTy& self) {
    void *p = c.data().getPerIterAlloc().allocate(sizeof(LocalState));
    new (p) LocalState(self, c.data().getPerIterAlloc());
    c.setLocalState(p, false);
  }
  void dealloc(GaloisRuntime::UserContextAccess<T>& c) {
    bool dummy;
    LocalState *p = reinterpret_cast<LocalState*>(c.data().getLocalState(dummy));
    p->~LocalState();
  }
  void save(GaloisRuntime::UserContextAccess<T>& c, void*& localState) { 
    bool dummy;
    localState = c.data().getLocalState(dummy);
  }
  void restore(GaloisRuntime::UserContextAccess<T>& c, void* localState) { 
    c.setLocalState(localState, true);
  }
};

template<typename T>
struct has_break_fn {
  typedef char yes[1];
  typedef char no[2];
  template<typename C> static yes& test(typename C::BreakFn*);
  template<typename> static no& test(...);
  static const bool value = sizeof(test<T>(0)) == sizeof(yes);
};

template<typename FunctionTy,typename Enable=void>
struct BreakManager {
  BreakManager(const FunctionTy&) { }
  bool checkBreak() { return false; }
};

template<typename FunctionTy>
class BreakManager<FunctionTy,typename boost::enable_if<has_break_fn<FunctionTy> >::type> {
  GBarrier barrier;
  LL::CacheLineStorage<volatile long> done;
  typename FunctionTy::BreakFn breakFn;

public:
  BreakManager(const FunctionTy& fn): breakFn(fn) { 
    int numActive = (int) Galois::getActiveThreads();
    barrier.reinit(numActive);
  }

  bool checkBreak() {
    runAllLoopExitHandlers();
    if (LL::getTID() == 0)
      done.data = breakFn();
    barrier.wait();
    return done.data;
  }
};

template<typename T>
struct DNewItem { 
  T val;
  unsigned long parent;
  unsigned count;

  DNewItem(const T& _val, unsigned long _parent, unsigned _count): val(_val), parent(_parent), count(_count) { }

  bool operator<(const DNewItem<T>& o) const {
    if (parent < o.parent)
      return true;
    else if (parent == o.parent)
      return count < o.count;
    else
      return false;
  }

  bool operator==(const DNewItem<T>& o) const {
    return parent == o.parent && count == o.count;
  }

  bool operator!=(const DNewItem<T>& o) const {
    return !(*this == o);
  }

  struct GetFirst: public std::unary_function<DNewItem<T>,const T&> {
    const T& operator()(const DNewItem<T>& x) const {
      return x.val;
    }
  };
};

template<typename InputIteratorTy>
void safe_advance(InputIteratorTy& it, size_t d, size_t& cur, size_t dist) {
  if (d + cur >= dist) {
    d = dist - cur;
  }
  std::advance(it, d);
  cur += d;
}

//! Wrapper around WorkList::ChunkedFIFO to allow peek() and empty() and still have FIFO order
template<int chunksize,typename T>
struct FIFO {
  WorkList::ChunkedFIFO<chunksize,T,false> m_data;
  WorkList::ChunkedLIFO<16,T,false> m_buffer;
  size_t m_size;

  FIFO(): m_size(0) { }

  ~FIFO() {
    boost::optional<T> p;
    while ((p = m_buffer.pop()))
      ;
    while ((p = m_data.pop()))
      ;
  }

  boost::optional<T> pop() {
    boost::optional<T> p;
    if ((p = m_buffer.pop()) || (p = m_data.pop())) {
      --m_size;
    }
    return p;
  }

  boost::optional<T> peek() {
    boost::optional<T> p;
    if ((p = m_buffer.pop())) {
      m_buffer.push(*p);
    } else if ((p = m_data.pop())) {
      m_buffer.push(*p);
    }
    return p;
  }

  void push(const T& val) {
    m_data.push(val);
    ++m_size;
  }

  size_t size() const {
    return m_size;
  }

  bool empty() const {
    return m_size == 0;
  }
};

template<typename> class DMergeManagerBase;
template<typename,typename> class DMergeManager;

//! Thread-local data for merging and implementations specialized for 
//! ordered and unordered implementations.
template<typename OptionsTy>
class DMergeLocal: private boost::noncopyable {
  template<typename> friend class DMergeManagerBase;
  template<typename,typename> friend class DMergeManager;

  typedef typename OptionsTy::T T;
  typedef typename OptionsTy::CompareTy CompareTy;

  struct HeapCompare {
    const CompareTy& comp;
    explicit HeapCompare(const CompareTy& c): comp(c) { }
    bool operator()(const T& a, const T& b) const {
      // reverse sense to get least items out of std::priority_queue
      return comp(b, a);
    }
  };

  typedef DItem<T> Item;
  typedef DNewItem<T> NewItem;
  typedef std::vector<NewItem,typename Galois::PerIterAllocTy::rebind<NewItem>::other> NewItemsTy;
  typedef FIFO<1024,Item> ReserveTy;

  typedef std::vector<T,typename Galois::PerIterAllocTy::rebind<T>::other> PQ;

  Galois::IterAllocBaseTy heap;
  Galois::PerIterAllocTy alloc;
  size_t window;
  size_t delta;
  size_t committed;
  size_t iterations;
  size_t aborted;
  NewItemsTy newItems;
  ReserveTy reserve;

  // For ordered execution
  PQ newReserve;
  boost::optional<T> mostElement;
  boost::optional<T> windowElement;

  // For id based execution
  size_t minId;
  size_t maxId;

  // For general execution
  size_t size;

public:
  DMergeLocal(): alloc(&heap), newItems(alloc), newReserve(alloc) { 
    resetStats(); 
  }

  ~DMergeLocal() {
    // itemPoolReset();
  }

private:
  //! Update min and max from sorted iterator
  template<typename BiIteratorTy>
  void initialLimits(BiIteratorTy ii, BiIteratorTy ei) {
    minId = std::numeric_limits<size_t>::max();
    maxId = std::numeric_limits<size_t>::min();
    mostElement = windowElement = boost::optional<T>();

    if (ii != ei) {
      if (ii + 1 == ei) {
        minId = maxId = ii->parent;
        mostElement = boost::optional<T>(ii->val);
      } else {
        minId = ii->parent;
        maxId = (ei-1)->parent;
        mostElement = boost::optional<T>(ei[-1].val);
      }
    }
  }

  template<typename WL>
  void nextWindowDispatch(WL* wl, const OptionsTy& options, UnorderedTag) {
    window += delta;
    boost::optional<Item> p;
    while ((p = reserve.peek())) {
      if (p->id >= window)
        break;
      wl->push(*p);
      reserve.pop();
    }
  }

  template<typename WL>
  void nextWindowDispatch(WL* wl, const OptionsTy& options, OrderedTag) {
    orderedUpdateDispatch<false>(wl, options.comp, 0);
  }

  template<typename WL>
  void updateWindowElement(WL* wl, const CompareTy& comp, size_t count) {
    orderedUpdateDispatch<true>(wl, comp, count);
  }

  //! Common functionality for (1) getting the next N-1 elements and setting windowElement
  //! to the nth element and (2) getting the next elements < windowElement.
  template<bool updateWE,typename WL>
  void orderedUpdateDispatch(WL* wl, const CompareTy& comp, size_t count) {
    // count = 0 is a special signal to not do anything
    if (updateWE && count == 0)
      return;

    if (updateWE) {
      size_t available = reserve.size() + newReserve.size();
      // No more reserve but what should we propose for windowElement? As with
      // distributeNewWork, this is a little tricky. Proposing nothing does not
      // work because our proposal must be at least as large as any element we
      // add to wl, and for progress, the element must be larger than at least
      // one element in the reserve. Here, we use the simplest solution which
      // is mostElement. 

      // TODO improve this
      if (available < count) {
        windowElement = mostElement;
        return;
      }
      count = std::min(count, available);
    }

    size_t c = 0;
    while (true) {
      boost::optional<Item> p1 = reserve.peek();
      boost::optional<T> p2 = peekNewReserve();

      bool fromReserve;
      if (p1 && p2)
        fromReserve = comp(p1->val, *p2);
      else if (!p1 && !p2)
        break;
      else
        fromReserve = p1;

      T* val = (fromReserve) ? &p1->val : &*p2;

      // When there is no mostElement or windowElement, the reserve should be
      // empty as well.
      assert(mostElement && windowElement);

      if (!comp(*val, *mostElement))
        break;
      if (!updateWE && !comp(*val, *windowElement))
        break;
      if (updateWE && ++c >= count) {
        windowElement = boost::optional<T>(*val);
        break;
      }
      
      wl->push(Item(*val, 0));

      if (fromReserve)
        reserve.pop();
      else
        popNewReserve(comp);
    }
  }

  template<typename InputIteratorTy,typename WL,typename NewTy>
  void copyInDispatch(InputIteratorTy ii, InputIteratorTy ei, size_t dist, WL* wl, NewTy&, unsigned numActive, const CompareTy& comp, UnorderedTag) {
    unsigned int tid = LL::getTID();
    size_t cur = 0;
    size_t k = 0;
    safe_advance(ii, tid, cur, dist);
    while (ii != ei) {
      unsigned long id = k * numActive + tid;
      if (id < window)
        wl->push(Item(*ii, id));
      else
        break;
      ++k;
      safe_advance(ii, numActive, cur, dist);
    }
    
    while (ii != ei) {
      unsigned long id = k * numActive + tid;
      reserve.push(Item(*ii, id));
      ++k;
      safe_advance(ii, numActive, cur, dist);
    }
  }

  template<typename InputIteratorTy,typename WL,typename NewTy>
  void copyInDispatch(InputIteratorTy ii, InputIteratorTy ei, size_t dist, WL* wl, NewTy& new_, unsigned numActive, const CompareTy& comp, OrderedTag) {
    assert(emptyReserve());

    unsigned int tid = LL::getTID();
    size_t cur = 0;
    safe_advance(ii, tid, cur, dist);
    while (ii != ei) {
      if (windowElement && !comp(*ii, *windowElement))
        break;
      wl->push(Item(*ii, 0));
      safe_advance(ii, numActive, cur, dist);
    }

    while (ii != ei) {
      if (mostElement && !comp(*ii, *mostElement))
        break;
      reserve.push(Item(*ii, 0));
      safe_advance(ii, numActive, cur, dist);
    }

    while (ii != ei) {
      new_.push(NewItem(*ii, 0, 1));
      safe_advance(ii, numActive, cur, dist);
    }
  }

  void initialWindow(size_t w) {
    window = delta = w;
  }

  void receiveLimits(const DMergeLocal<OptionsTy>& other) {
    minId = other.minId;
    maxId = other.maxId;
    mostElement = other.mostElement;
    windowElement = other.windowElement;
    size = other.size;
  }

  void reduceLimits(const DMergeLocal<OptionsTy>& other, const CompareTy& comp) {
    minId = std::min(other.minId, minId);
    maxId = std::max(other.maxId, maxId);
    size += other.size;

    if (!mostElement) {
      mostElement = other.mostElement;
    } else if (other.mostElement && comp(*mostElement, *other.mostElement)) {
      mostElement = other.mostElement;
    }

    if (!windowElement) {
      windowElement = other.windowElement;
    } else if (other.windowElement && comp(*windowElement, *other.windowElement)) {
      windowElement = other.windowElement;
    }
  }

  void popNewReserve(const CompareTy& comp) {
    std::pop_heap(newReserve.begin(), newReserve.end(), HeapCompare(comp));
    newReserve.pop_back();
  }

  void pushNewReserve(const T& val, const CompareTy& comp) {
    newReserve.push_back(val);
    std::push_heap(newReserve.begin(), newReserve.end(), HeapCompare(comp));
  }

  boost::optional<T> peekNewReserve() {
    if (newReserve.empty())
      return boost::optional<T>();
    else
      return boost::optional<T>(newReserve.front());
  }
  
  template<typename InputIteratorTy,typename WL,typename NewTy>
  void copyIn(InputIteratorTy b, InputIteratorTy e, size_t dist, WL* wl, NewTy& new_, unsigned numActive, const CompareTy& comp) {
    copyInDispatch(b, e, dist, wl, new_, numActive, comp, typename OptionsTy::Tag());
  }

public:
  void clear() {
    // itemPoolReset();
    heap.clear();
  }

  void incrementIterations() {
    ++iterations;
  }

  void incrementCommitted() {
    ++committed;
  }

  void assertLimits(const T& val, const CompareTy& comp) {
    assert(!windowElement || comp(val, *windowElement));
    assert(!mostElement || comp(val, *mostElement));
  }

  template<typename WL>
  void nextWindow(WL* wl, const OptionsTy& options) {
    nextWindowDispatch(wl, options, typename OptionsTy::Tag());
  }

  void resetStats() {
    committed = iterations = aborted = 0;
  }

  bool emptyReserve() {
    return reserve.empty() && newReserve.empty();
  }
};

template<typename T>
struct has_id_fn {
  typedef char yes[1];
  typedef char no[2];
  template<typename C> static yes& test(typename C::IdFn*);
  template<typename> static no& test(...);
  static const bool value = sizeof(test<T>(0)) == sizeof(yes);
};

template<typename OptionsTy,typename Enable=void>
struct MergeTraits {
  static const bool value = false;
  static const int ChunkSize = 32;
  static const int MinDelta = ChunkSize * 40;
};

template<typename OptionsTy>
struct MergeTraits<OptionsTy,typename boost::enable_if_c<OptionsTy::useOrdered>::type> {
  static const bool value = true;
  struct DummyIdFn {
    typedef typename OptionsTy::T T;
    unsigned long operator()(const T& x) const { return 0; }
  };
  typedef DummyIdFn IdFn;
  static const int ChunkSize = 16;
  static const int MinDelta = 4;
};

template<typename OptionsTy>
struct MergeTraits<OptionsTy,typename boost::enable_if_c<has_id_fn<typename OptionsTy::Function1Ty>::value && !OptionsTy::useOrdered>::type> {
  static const bool value = true;
  typedef typename OptionsTy::Function1Ty::IdFn IdFn;
  static const int ChunkSize = 32;
  static const int MinDelta = ChunkSize * 40;
};


template<typename OptionsTy>
class DMergeManagerBase {
protected:
  typedef typename OptionsTy::T T;
  typedef typename OptionsTy::CompareTy CompareTy;
  typedef DItem<T> Item;
  typedef DNewItem<T> NewItem;
  typedef WorkList::dChunkedFIFO<MergeTraits<OptionsTy>::ChunkSize,NewItem> NewWork;
  typedef DMergeLocal<OptionsTy> MergeLocal;
  typedef typename MergeLocal::NewItemsTy NewItemsTy;
  typedef typename NewItemsTy::iterator NewItemsIterator;

  Galois::IterAllocBaseTy heap;
  Galois::PerIterAllocTy alloc;
  PerThreadStorage<MergeLocal> data;

  NewWork new_;
  unsigned numActive;

  void broadcastLimits(MergeLocal& mlocal, unsigned int tid) {
    for (unsigned i = 0; i < this->numActive; ++i) {
      if (i == tid) continue;
      MergeLocal& mother = *this->data.getRemote(i);
      mother.receiveLimits(mlocal);
    }
  }

  void reduceLimits(MergeLocal& mlocal, unsigned int tid, const CompareTy& comp) {
    for (unsigned i = 0; i < this->numActive; ++i) {
      if (i == tid) continue;
      MergeLocal& mother = *this->data.getRemote(i);
      mlocal.reduceLimits(mother, comp);
    }
  }

public:
  DMergeManagerBase(): alloc(&heap) {
    numActive = Galois::getActiveThreads();
  }

  ~DMergeManagerBase() {
    boost::optional<NewItem> p;
    assert(!(p = new_.pop()));
  }

  MergeLocal& get() {
    return *data.getLocal();
  }

  void calculateWindow(bool inner) {
    MergeLocal& mlocal = *data.getLocal();

    // Accumulate all threads' info
    size_t allcommitted = 0;
    size_t alliterations = 0;
    for (unsigned i = 0; i < numActive; ++i) {
      MergeLocal& mlocal = *data.getRemote(i);
      allcommitted += mlocal.committed;
      alliterations += mlocal.iterations;
    }

    const float target = 0.95;
    float commitRatio = alliterations > 0 ? allcommitted / (float) alliterations : 0.0;

    if (commitRatio >= target)
      mlocal.delta += mlocal.delta;
    else if (allcommitted == 0) // special case when we don't execute anything
      mlocal.delta += mlocal.delta;
    else
      mlocal.delta = commitRatio / target * mlocal.delta;

    if (!inner) {
      mlocal.delta = std::max(mlocal.delta, (size_t) MergeTraits<OptionsTy>::MinDelta);
    } else if (mlocal.delta < (size_t) MergeTraits<OptionsTy>::MinDelta) {
      // Try to get some new work instead of increasing window
      mlocal.delta = 0;
    }

    // Useful debugging info
    if (false) {
      if (LL::getTID() == 0) {
        LL::gDebug("DEBUG %d %.3f (%zu/%zu) window: %zu delta: %zu\n", 
            inner, commitRatio, allcommitted, alliterations, mlocal.window, mlocal.delta);
      }
    }
  }
};

//! Default implementation for merging
template<typename OptionsTy,typename Enable=void>
class DMergeManager: public DMergeManagerBase<OptionsTy> {
  typedef DMergeManagerBase<OptionsTy> Base;
  typedef typename Base::T T;
  typedef typename Base::Item Item;
  typedef typename Base::NewItem NewItem;
  typedef typename Base::MergeLocal MergeLocal;
  typedef typename Base::NewItemsTy NewItemsTy;
  typedef typename Base::NewItemsIterator NewItemsIterator;
  typedef typename Base::CompareTy CompareTy;

  struct GetNewItem: public std::unary_function<int,NewItemsTy&> {
    PerThreadStorage<MergeLocal>* base;
    GetNewItem() { }
    GetNewItem(PerThreadStorage<MergeLocal>* b): base(b) { }
    NewItemsTy& operator()(int i) const { return base->getRemote(i)->newItems; }
  };

  typedef boost::transform_iterator<GetNewItem, boost::counting_iterator<int> > MergeOuterIt;
  typedef typename Galois::ChooseStlTwoLevelIterator<MergeOuterIt, typename NewItemsTy::iterator>::type MergeIt;

  std::vector<NewItem,typename Galois::PerIterAllocTy::rebind<NewItem>::other> mergeBuf;
  std::vector<T,typename Galois::PerIterAllocTy::rebind<T>::other> distributeBuf;

  GBarrier barrier[4];

  bool merge(int begin, int end) {
    if (begin == end)
      return false;
    else if (begin + 1 == end)
      return !this->data.getRemote(begin)->newItems.empty();
    
    bool retval = false;
    int mid = (end - begin) / 2 + begin;
    retval |= merge(begin, mid);
    retval |= merge(mid, end);

    MergeOuterIt bbegin(boost::make_counting_iterator(begin), GetNewItem(&this->data));
    MergeOuterIt mmid(boost::make_counting_iterator(mid), GetNewItem(&this->data));
    MergeOuterIt eend(boost::make_counting_iterator(end), GetNewItem(&this->data));
    // MergeIt aa(bbegin, mmid), ea(mmid, mmid);
    // MergeIt bb(mmid, eend), eb(eend, eend);
    // MergeIt cc(bbegin, eend), ec(eend, eend);
    MergeIt aa = Galois::stl_two_level_begin(bbegin, mmid);
    MergeIt ea = Galois::stl_two_level_end(bbegin, mmid);
    MergeIt bb = Galois::stl_two_level_begin(mmid, eend);
    MergeIt eb = Galois::stl_two_level_end(mmid, eend);
    MergeIt cc = Galois::stl_two_level_begin(bbegin, eend);
    MergeIt ec = Galois::stl_two_level_end(bbegin, eend);

    while (aa != ea && bb != eb) {
      if (*aa < *bb)
        mergeBuf.push_back(*aa++);
      else
        mergeBuf.push_back(*bb++);
    }

    for (; aa != ea; ++aa)
      mergeBuf.push_back(*aa);

    for (; bb != eb; ++bb)
      mergeBuf.push_back(*bb);

    for (NewItemsIterator ii = mergeBuf.begin(), ei = mergeBuf.end(); ii != ei; ++ii) 
      *cc++ = *ii; 

    mergeBuf.clear();

    assert(cc == ec);

    return retval;
  }

  //! Slightly complicated reindexing to separate out continuous elements in InputIterator
  template<typename InputIteratorTy>
  void redistribute(InputIteratorTy ii, InputIteratorTy ei, size_t dist) {
    unsigned int tid = LL::getTID();
    //const size_t numBlocks = 1 << 7;
    //const size_t mask = numBlocks - 1;
    //size_t blockSize = dist / numBlocks; // round down
    MergeLocal& mlocal = *this->data.getLocal();
    //size_t blockSize = std::max((size_t) (0.9*minfo.delta), (size_t) 1);
    size_t blockSize = mlocal.delta;
    size_t numBlocks = dist / blockSize;
    
    size_t cur = 0;
    safe_advance(ii, tid, cur, dist);
    while (ii != ei) {
      unsigned long id;
      if (cur < blockSize * numBlocks)
        //id = (cur & mask) * blockSize + (cur / numBlocks);
        id = (cur % numBlocks) * blockSize + (cur / numBlocks);
      else
        id = cur;
      distributeBuf[id] = *ii;
      safe_advance(ii, this->numActive, cur, dist);
    }
  }

  template<typename InputIteratorTy,typename WL>
  void distribute(InputIteratorTy ii, InputIteratorTy ei, size_t dist, WL* wl) {
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = *this->data.getLocal();
    mlocal.initialWindow(std::max(dist / 100, (size_t) MergeTraits<OptionsTy>::MinDelta));
    if (true) {
      // Renumber to avoid pathological cases
      if (tid == 0) {
        distributeBuf.resize(dist);
      }
      barrier[0].wait();
      redistribute(ii, ei, dist);
      barrier[1].wait();
      mlocal.copyIn(distributeBuf.begin(), distributeBuf.end(), dist, wl, this->new_, this->numActive, CompareTy());
    } else {
      mlocal.copyIn(ii, ei, dist, wl, this->new_, this->numActive, CompareTy());
    }
  }

  template<typename WL>
  void parallelSort(WL* wl) {
    MergeLocal& mlocal = *this->data.getLocal();

    mlocal.newItems.clear();
    boost::optional<NewItem> p;
    while ((p = this->new_.pop())) {
      mlocal.newItems.push_back(*p);
    }

    std::sort(mlocal.newItems.begin(), mlocal.newItems.end());
    mlocal.size = mlocal.newItems.size();
    
    barrier[2].wait();

    unsigned tid = LL::getTID();
    if (tid == 0) {
      this->reduceLimits(mlocal, tid, CompareTy());
      mergeBuf.reserve(mlocal.size);
      this->broadcastLimits(mlocal, tid);
      merge(0, this->numActive);
    }

    barrier[3].wait();

    MergeOuterIt bbegin(boost::make_counting_iterator(0), GetNewItem(&this->data));
    MergeOuterIt eend(boost::make_counting_iterator((int) this->numActive), GetNewItem(&this->data));
    MergeIt ii = Galois::stl_two_level_begin(bbegin, eend);
    MergeIt ei = Galois::stl_two_level_end(eend, eend);

    distribute(boost::make_transform_iterator(ii, typename Base::NewItem::GetFirst()),
        boost::make_transform_iterator(ei, typename Base::NewItem::GetFirst()),
        mlocal.size, wl);
  }

  template<typename WL>
  void serialSort(WL* wl) {
    this->new_.flush();

    barrier[2].wait();
    
    if (LL::getTID() == 0) {
      mergeBuf.clear();
      boost::optional<NewItem> p;
      while ((p = this->new_.pop())) {
        mergeBuf.push_back(*p);
      }

      std::sort(mergeBuf.begin(), mergeBuf.end());

      printf("DEBUG R %zd\n", mergeBuf.size());
    }

    barrier[3].wait();

    distribute(boost::make_transform_iterator(mergeBuf.begin(), typename NewItem::GetFirst()),
        boost::make_transform_iterator(mergeBuf.end(), typename NewItem::GetFirst()),
        mergeBuf.size(), wl);
  }

public:
  DMergeManager(const OptionsTy& o): mergeBuf(this->alloc), distributeBuf(this->alloc) {
    for (unsigned i = 0; i < sizeof(barrier)/sizeof(*barrier); ++i)
      barrier[i].reinit(this->numActive);
  }

  template<typename InputIteratorTy>
  void presort(InputIteratorTy ii, InputIteratorTy ei) { }

  template<typename InputIteratorTy, typename WL>
  void addInitialWork(InputIteratorTy b, InputIteratorTy e, WL* wl) {
    distribute(b, e, std::distance(b, e), wl);
  }

  template<typename WL>
  void pushNew(const T& val, unsigned long parent, unsigned count, WL* wl, bool& hasNewWork, bool& nextCommit) {
    this->new_.push(NewItem(val, parent, count));
    hasNewWork = true;
  }

  template<typename WL>
  bool distributeNewWork(WL* wl) {
    if (true)
      parallelSort(wl);
    else
      serialSort(wl);
    return false;
  }

  template<typename WL>
  void prepareNextWindow(WL* wl) { }
};

//! Implementation of merging specialized for unordered algorithms with an id function and ordered algorithms
// TODO: For consistency should also have thread-local copies of comp and
// idFunction
template<typename OptionsTy>
class DMergeManager<OptionsTy,typename boost::enable_if<MergeTraits<OptionsTy> >::type>: public DMergeManagerBase<OptionsTy> {
  typedef DMergeManagerBase<OptionsTy> Base;
  typedef typename Base::T T;
  typedef typename Base::Item Item;
  typedef typename Base::NewItem NewItem;
  typedef typename Base::MergeLocal MergeLocal;
  typedef typename Base::NewItemsTy NewItemsTy;
  typedef typename Base::NewItemsIterator NewItemsIterator;
  typedef typename MergeTraits<OptionsTy>::IdFn IdFn;
  typedef typename Base::CompareTy CompareTy;

  struct CompareNewItems: public std::binary_function<NewItem,NewItem,bool> {
    const CompareTy& comp;
    CompareNewItems(const CompareTy& c): comp(c) { }
    bool operator()(const NewItem& a, const NewItem& b) const {
      return comp(a.val, b.val);
    }
  };

  std::vector<NewItem,typename Galois::PerIterAllocTy::rebind<NewItem>::other> mergeBuf;

  GBarrier barrier;
  IdFn idFunction;
  CompareTy comp;

public:
  DMergeManager(const OptionsTy& o): mergeBuf(this->alloc), comp(o.comp) {
    barrier.reinit(this->numActive);
  }

  template<typename InputIteratorTy, typename WL>
  void addInitialWork(InputIteratorTy ii, InputIteratorTy ei, WL* wl) {
    MergeLocal& mlocal = *this->data.getLocal();
    mlocal.copyIn(
        boost::make_transform_iterator(mergeBuf.begin(), typename Base::NewItem::GetFirst()),
        boost::make_transform_iterator(mergeBuf.end(), typename Base::NewItem::GetFirst()),
        mergeBuf.size(), wl, this->new_, this->numActive, comp);
  }

  template<typename InputIteratorTy>
  void presort(InputIteratorTy ii, InputIteratorTy ei) {
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = *this->data.getLocal();
    size_t dist = std::distance(ii, ei);

    // Ordered algorithms generally have less available parallelism, so start
    // window size out small
    size_t window;
    
    if (OptionsTy::useOrdered)
      window = std::min((size_t) this->numActive, dist);

    assert(mergeBuf.empty());

    mergeBuf.reserve(dist);
    for (; ii != ei; ++ii)
      mergeBuf.push_back(NewItem(*ii, idFunction(*ii), 1));

    if (OptionsTy::useOrdered)
      Galois::ParallelSTL::sort(mergeBuf.begin(), mergeBuf.end(), CompareNewItems(comp));
    else
      Galois::ParallelSTL::sort(mergeBuf.begin(), mergeBuf.end());

    mlocal.initialLimits(mergeBuf.begin(), mergeBuf.end());
    if (OptionsTy::useOrdered) {
      if (window)
        mlocal.windowElement = boost::optional<T>(mergeBuf[window-1].val);
    }
    
    this->broadcastLimits(mlocal, tid);

    if (!OptionsTy::useOrdered)
      window = std::max((mlocal.maxId - mlocal.minId) / 100, (size_t) MergeTraits<OptionsTy>::MinDelta) + mlocal.minId;

    for (unsigned i = 0; i < this->numActive; ++i) {
      MergeLocal& mother = *this->data.getRemote(i);
      mother.initialWindow(window);
    }
  }

  template<typename WL>
  void pushNew(const T& val, unsigned long parent, unsigned count, WL* wl, bool& hasNewWork, bool& nextCommit) {
    if (!OptionsTy::useOrdered) {
      this->new_.push(NewItem(val, idFunction(val), 1));
      hasNewWork = true;
      return;
    }

    // OptionsTy::useOrdered is true below

    MergeLocal& mlocal = *this->data.getLocal();

    // NB: Tricky conditions. If we can not definitively place an item, it must
    // go into the current wl.
    if (mlocal.mostElement && !comp(val, *mlocal.mostElement)) {
      this->new_.push(NewItem(val, idFunction(val), 1));
      hasNewWork = true;
    } else if (mlocal.mostElement && mlocal.windowElement && !comp(val, *mlocal.windowElement)) {
      mlocal.pushNewReserve(val, comp);
    } else {
      // TODO: account for this work in calculateWindow
      wl->push(Item(val, 0));
      nextCommit = true;
    }
  }

  template<typename WL>
  bool distributeNewWork(WL* wl) {
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = *this->data.getLocal();

    assert(mlocal.emptyReserve());

    mlocal.newItems.clear();
    boost::optional<NewItem> p;
    while ((p = this->new_.pop()))
      mlocal.newItems.push_back(*p);

    if (OptionsTy::useOrdered)
      std::sort(mlocal.newItems.begin(), mlocal.newItems.end(), CompareNewItems(comp));
    else
      std::sort(mlocal.newItems.begin(), mlocal.newItems.end());

    NewItemsIterator ii = mlocal.newItems.begin(), ei = mlocal.newItems.end();
    mlocal.initialLimits(ii, ei);

    if (OptionsTy::useOrdered) {
      // Smallest useful delta is 2 because windowElement is not included into
      // current workset
      size_t w = std::min(std::max(mlocal.delta / this->numActive, (size_t) 2), mlocal.newItems.size());
      if (w)
        mlocal.windowElement = boost::optional<T>(mlocal.newItems[w-1].val);
    }

    barrier.wait();
    
    if (tid == 0) {
      this->reduceLimits(mlocal, tid, comp);
      this->broadcastLimits(mlocal, tid);
    }

    barrier.wait();

    bool retval = false;

    if (OptionsTy::useOrdered) {
      mlocal.initialWindow(this->numActive);
      
      assert(ii == ei || (mlocal.windowElement && mlocal.mostElement));
      assert((!mlocal.windowElement && !mlocal.mostElement) || !comp(*mlocal.mostElement, *mlocal.windowElement));

      // No new items; we just have the most element X from the previous round.
      // The most and window elements are exclusive of the range that they
      // define; there is no most or window element that includes X. The
      // easiest solution is to not use most or window elements for the next
      // round, but the downside is that we will never return to windowed execution.

      // TODO: improve this
      if (mlocal.windowElement && mlocal.mostElement && !comp(*mlocal.windowElement, *mlocal.mostElement)) {
        mlocal.windowElement = mlocal.mostElement = boost::optional<T>();
        for (; ii != ei; ++ii) {
          wl->push(Item(ii->val, 0));
        }
      }

      for (; ii != ei; ++ii) {
        if (!comp(ii->val, *mlocal.windowElement))
          break; 
        wl->push(Item(ii->val, 0));
      }

      for (; ii != ei; ++ii) {
        if (!comp(ii->val, *mlocal.mostElement))
          break;
        mlocal.reserve.push(Item(ii->val, 0));
      }

      for (; ii != ei; ++ii) {
        retval = true;
        this->new_.push(NewItem(ii->val, idFunction(ii->val), 1));
      }
    } else {
      mlocal.initialWindow(std::max((mlocal.maxId - mlocal.minId) / 100, (size_t) MergeTraits<OptionsTy>::MinDelta) + mlocal.minId);

      for (; ii != ei; ++ii) {
        unsigned long id = ii->parent;
        if (id < mlocal.window)
          wl->push(Item(ii->val, id));
        else
          break;
      }

      for (; ii != ei; ++ii) {
        unsigned long id = ii->parent;
        mlocal.reserve.push(Item(ii->val, id));
      }
    }

    return retval;
  }

  template<typename WL>
  void prepareNextWindow(WL* wl) { 
    if (!OptionsTy::useOrdered)
      return;
    
    unsigned int tid = LL::getTID();
    MergeLocal& mlocal = *this->data.getLocal();
    size_t w = 0;
    // Convert non-zero deltas into per thread counts
    if (mlocal.delta) {
      if (mlocal.delta < this->numActive)
        w = tid < mlocal.delta ? 1 : 0;
      else 
        w = mlocal.delta / this->numActive;
      w++; // exclusive end point
    }
    mlocal.updateWindowElement(wl, comp, w);

    barrier.wait();

    if (tid == 0) {
      this->reduceLimits(mlocal, tid, comp);
      this->broadcastLimits(mlocal, tid);
    }
  }
};

template<typename OptionsTy>
class Executor {
  typedef typename OptionsTy::T value_type;
  typedef DItem<value_type> Item;
  typedef DNewItem<value_type> NewItem;
  typedef DMergeManager<OptionsTy> MergeManager;
  typedef DMergeLocal<OptionsTy> MergeLocal;

  typedef typename OptionsTy::DetContext DetContext;

  typedef WorkList::dChunkedFIFO<MergeTraits<OptionsTy>::ChunkSize,Item> WL;
  typedef WorkList::dChunkedFIFO<MergeTraits<OptionsTy>::ChunkSize,DetContext> PendingWork;
  typedef WorkList::ChunkedFIFO<MergeTraits<OptionsTy>::ChunkSize,DetContext,false> LocalPendingWork;
  static const bool useLocalState = has_local_state<typename OptionsTy::Function1Ty>::value;

  // Truly thread-local
  struct ThreadLocalData: private boost::noncopyable {
    OptionsTy options;
    LocalPendingWork localPending;
    GaloisRuntime::UserContextAccess<value_type> facing;
    LoopStatistics<OptionsTy::needsStats> stat;
    WL* wlcur;
    WL* wlnext;
    size_t rounds;
    size_t outerRounds;
    bool hasNewWork;
    ThreadLocalData(const OptionsTy& o, const char* loopname): options(o), stat(loopname), rounds(0), outerRounds(0) { }
  };

  const OptionsTy& origOptions;
  MergeManager mergeManager;
  const char* loopname;
  BreakManager<typename OptionsTy::Function1Ty> breakManager;
  GBarrier barrier[4];
  WL worklists[2];
  StateManager<value_type,typename OptionsTy::Function1Ty> stateManager;
  PendingWork pending;
  // ContextPool<DeterministicRuntimeContext> contextPool;
  LL::CacheLineStorage<volatile long> innerDone;
  LL::CacheLineStorage<volatile long> outerDone;
  LL::CacheLineStorage<volatile long> hasNewWork;
  int numActive;

  bool pendingLoop(ThreadLocalData& tld);
  bool commitLoop(ThreadLocalData& tld);
  void go();

public:
  Executor(const OptionsTy& o, const char* ln):
    origOptions(o), mergeManager(o), loopname(ln), breakManager(o.fn1)
  { 
    numActive = (int) Galois::getActiveThreads();
    for (unsigned i = 0; i < sizeof(barrier)/sizeof(*barrier); ++i)
      barrier[i].reinit(numActive);
    if (OptionsTy::needsBreak && !has_break_fn<typename OptionsTy::Function1Ty>::value) {
      GALOIS_ERROR(true, "need to use break function to break loop");
    }
  }

  template<typename RangeTy>
  void AddInitialWork(RangeTy range) {
    mergeManager.addInitialWork(range.begin(), range.end(), &worklists[1]);
  }

  template<typename IterTy>
  void presort(IterTy ii, IterTy ei) {
    mergeManager.presort(ii, ei);
  }

  void operator()() {
    go();
  }
};

template<typename OptionsTy>
void Executor<OptionsTy>::go() {
  ThreadLocalData tld(origOptions, loopname);
  MergeLocal& mlocal = mergeManager.get();
  tld.wlcur = &worklists[0];
  tld.wlnext = &worklists[1];

  // copyIn for ordered algorithms adds at least one initial new item
  tld.hasNewWork = OptionsTy::useOrdered ? true : false;

  while (true) {
    ++tld.outerRounds;

    while (true) {
      ++tld.rounds;

      std::swap(tld.wlcur, tld.wlnext);
      setPending(PENDING);
      bool nextPending = pendingLoop(tld);
      innerDone.data = true;

      barrier[1].wait();

      setPending(COMMITTING);
      bool nextCommit = commitLoop(tld);
      outerDone.data = true;
      if (nextPending || nextCommit)
        innerDone.data = false;

      barrier[2].wait();

      // contextPool.commitAll();
      // mlocal.itemPoolReset();

      if (innerDone.data)
        break;

      mergeManager.calculateWindow(true);
      mergeManager.prepareNextWindow(tld.wlnext);

      barrier[0].wait();

      mlocal.nextWindow(tld.wlnext, tld.options);
      mlocal.resetStats();
    }

    if (!mlocal.emptyReserve())
      outerDone.data = false;

    if (tld.hasNewWork)
      hasNewWork.data = true;

    if (breakManager.checkBreak())
      break;

    mergeManager.calculateWindow(false);
    mergeManager.prepareNextWindow(tld.wlnext);

    barrier[3].wait();

    if (outerDone.data) {
      if (!OptionsTy::needsPush)
        break;
      if (!hasNewWork.data) // (1)
        break;
      tld.hasNewWork = mergeManager.distributeNewWork(tld.wlnext);
      // NB: assumes that distributeNewWork has a barrier otherwise checking at (1) is erroneous
      hasNewWork.data = false;
    } else {
      mlocal.nextWindow(tld.wlnext, tld.options);
    }

    mlocal.resetStats();
  }

  setPending(NON_DET);

  mlocal.clear(); // parallelize clean up too

  if (OptionsTy::needsStats) {
    if (LL::getTID() == 0) {
      reportStat(loopname, "RoundsExecuted", tld.rounds);
      reportStat(loopname, "OuterRoundsExecuted", tld.outerRounds);
    }
  }
}

template<typename OptionsTy>
bool Executor<OptionsTy>::pendingLoop(ThreadLocalData& tld)
{
  MergeLocal& mlocal = mergeManager.get();
  bool retval = false;
  boost::optional<Item> p;
  while ((p = tld.wlcur->pop())) {
    // Use a new context for each item.
    // There is a race when reusing between aborted iterations.
    // DeterministicRuntimeContext* cnx = contextPool.next();

    DetContext* ctx = NULL;
    if (useLocalState) {
      ctx = tld.localPending.push(tld.options.make_context(*p));
    } else {
      ctx = pending.push(tld.options.make_context(*p));
    }

    assert(ctx != NULL);

    mlocal.incrementIterations();
    bool commit = true;
    // cnx->set_id(p->id);
    if (OptionsTy::useOrdered) {
      // cnx->set_comp(&options.comp);
      // cnx->set_comp_data(mlocal.itemPoolPush(p->item));
      // mlocal.assertLimits(p->item, options.comp);
      mlocal.assertLimits(ctx->item.val, tld.options.comp);
    }

    ctx->start_iteration();
    tld.stat.inc_iterations();
    setThreadContext(ctx);

    stateManager.alloc(tld.facing, tld.options.fn1);
    int result = 0;
#if GALOIS_USE_EXCEPTION_HANDLER
    try {
      tld.options.fn1(ctx->item.val, tld.facing.data());
    } catch (ConflictFlag flag) {
      clearConflictLock();
      result = flag;
    }
#else
    if ((result = setjmp(hackjmp)) == 0) {
      tld.options.fn1(ctx->item.val, tld.facing.data());
    }
#endif
    switch (result) {
      case 0: 
      case REACHED_FAILSAFE: break;
      case CONFLICT: commit = false; break;
      default: assert(0 && "Unknown conflict flag"); abort(); break;
    }

    if (ForEachTraits<typename OptionsTy::Function1Ty>::NeedsPIA && !useLocalState)
      tld.facing.resetAlloc();

    // if (!commit) {
      // stateManager.dealloc(tld.facing); 
      // // tld.wlnext->push(*p);
      // assert (!p->is_ready ());
      // tld.stat.inc_conflicts();
      // retval = true;
    // }

    if (commit) { 
      if (useLocalState) {
        stateManager.save(tld.facing, ctx->item.localState);
      }
    } else {
      retval = true; 
    }

  }

  return retval;
}

template<typename OptionsTy>
bool Executor<OptionsTy>::commitLoop(ThreadLocalData& tld) 
{
  bool retval = false;
  MergeLocal& mlocal = mergeManager.get();

  size_t ncommits = 0;
  size_t niter = 0;

  boost::optional<DetContext> p;
  while ((p = (useLocalState) ? tld.localPending.pop() : pending.pop())) {
    ++niter;
    bool commit = true;
    // Can skip this check in prefix repeating computations but eagerly aborting
    // seems more efficient
    DetContext* ctx = &*p;
    if (ctx->notReady())
      commit = false;

    if (commit) {
      setThreadContext(ctx);
      stateManager.restore(tld.facing, ctx->item.localState);
      int result = 0;
#if GALOIS_USE_EXCEPTION_HANDLER
      try {
        tld.options.fn2(ctx->item.val, tld.facing.data());
      } catch (ConflictFlag flag) {
        clearConflictLock();
        result = flag;
      }
#else
      if ((result = setjmp(hackjmp)) == 0) {
        tld.options.fn2(ctx->item.val, tld.facing.data());
      }
#endif
      switch (result) {
        case 0: break;
        case CONFLICT: commit = false; break;
        default: assert(0 && "Unknown conflict flag"); abort(); break;
      }

    }

    stateManager.dealloc(tld.facing);
    
    if (commit) {
      ++ncommits;
      mlocal.incrementCommitted();
      if (ForEachTraits<typename OptionsTy::Function2Ty>::NeedsPush) {
        unsigned long parent = ctx->item.id;
        typedef typename UserContextAccess<value_type>::PushBufferTy::iterator iterator;
        unsigned count = 0;
        for (iterator ii = tld.facing.getPushBuffer().begin(), 
            ei = tld.facing.getPushBuffer().end(); ii != ei; ++ii) {
          mergeManager.pushNew(*ii, parent, ++count, tld.wlnext, tld.hasNewWork, retval);
          if (count == 0) {
            assert(0 && "Counter overflow");
            abort();
          }
        }
      }
      assert(ForEachTraits<typename OptionsTy::Function2Ty>::NeedsPush
          || tld.facing.getPushBuffer().begin() == tld.facing.getPushBuffer().end());
    } else {
      // p->cnx = NULL;
      //if (useLocalState) p->localState = NULL;
      if (useLocalState) { ctx->item.localState = NULL; }
      tld.wlnext->push(ctx->item);
      tld.stat.inc_conflicts();
      retval = true;
    }

    if (commit) {
      ctx->commit_iteration();
    } else {
      ctx->cancel_iteration();
    }


    if (ForEachTraits<typename OptionsTy::Function2Ty>::NeedsPIA && !useLocalState)
      tld.facing.resetAlloc();

    tld.facing.resetPushBuffer();


  } // end while

  if (ForEachTraits<typename OptionsTy::Function2Ty>::NeedsPIA && useLocalState)
    tld.facing.resetAlloc();

  setThreadContext(0);
  if (false && LL::getTID() == 0) {
    GALOIS_DEBUG("niter = %zd, ncommits = %zd", niter, ncommits);
  }
  return retval;
}


} // end namespace DeterministicImpl

template<typename InitTy, typename WorkTy>
static inline void for_each_det_impl(InitTy& init, WorkTy& W) {
  W.presort(init.range.begin(), init.range.end());

  assert(!inGaloisForEach);

  inGaloisForEach = true;
  RunCommand w[4] = {Config::ref(init), 
		     Config::ref(getSystemBarrier()),
		     Config::ref(W),
		     Config::ref(getSystemBarrier())};
  getSystemThreadPool().run(&w[0], &w[4]);
  runAllLoopExitHandlers();
  inGaloisForEach = false;
}


//! Ordered set iteration using deterministic executor
template<typename IterTy, typename ComparatorTy, typename NhFunc, typename OpFunc>
static inline void for_each_ordered_2p(IterTy b, IterTy e, ComparatorTy comp, NhFunc f1, OpFunc f2, const char* loopname) {
  typedef GaloisRuntime::StandardRange<IterTy> Range;
  typedef typename Range::value_type T;
  typedef GaloisRuntime::DeterministicImpl::OrderedOptions<T,NhFunc,OpFunc,ComparatorTy> OptionsTy;
  typedef GaloisRuntime::DeterministicImpl::Executor<OptionsTy> WorkTy;

  OptionsTy options(f1, f2, comp);
  WorkTy W(options, loopname);
  GaloisRuntime::Initializer<Range, WorkTy> init(Range(b, e), W);
  for_each_det_impl(init, W);
}


} // end namespace GaloisRuntime

namespace Galois {

/**
 * Deterministic execution with prefix operator.
 * The prefix of the operator should be exactly the same as the operator
 * but with execution returning at the failsafe point. The operator
 * should conform to a standard Galois unordered set operator {@link for_each()}.
 *
 * @param b begining of range of initial items
 * @param e end of range of initial items
 * @param prefix prefix of operator
 * @param fn operator
 * @param loopname string to identity loop in statistics output
 */
template<typename IterTy, typename Function1Ty, typename Function2Ty>
static inline void for_each_det(IterTy b, IterTy e, Function1Ty prefix, Function2Ty fn, const char* loopname = 0) {
  typedef GaloisRuntime::StandardRange<IterTy> Range;
  typedef typename Range::value_type T;
  typedef GaloisRuntime::DeterministicImpl::UnorderedOptions<T,Function1Ty,Function2Ty> OptionsTy;
  typedef GaloisRuntime::DeterministicImpl::Executor<OptionsTy> WorkTy;

  OptionsTy options(prefix, fn);
  WorkTy W(options, loopname);
  GaloisRuntime::Initializer<Range, WorkTy> init(Range(b, e), W);
  GaloisRuntime::for_each_det_impl(init, W);
}

/**
 * Deterministic execution with prefix operator.
 * The prefix of the operator should be exactly the same as the operator
 * but with execution returning at the failsafe point. The operator
 * should conform to a standard Galois unordered set operator {@link for_each()}.
 *
 * @param i initial item
 * @param prefix prefix of operator
 * @param fn operator
 * @param loopname string to identity loop in statistics output
 */
template<typename T, typename Function1Ty, typename Function2Ty>
static inline void for_each_det(T i, Function1Ty prefix, Function2Ty fn, const char* loopname = 0) {
  T wl[1] = { i };
  for_each_det(&wl[0], &wl[1], prefix, fn, loopname);
}

/**
 * Deterministic execution with single operator.
 * The operator fn is used both for the prefix computation and for the
 * continuation of computation, c.f., the prefix operator version which
 * uses two different functions. The operator can distinguish between
 * the two uses by querying {@link UserContext.getLocalState()}.
 *
 * @param b begining of range of initial items
 * @param e end of range of initial items
 * @param fn operator
 * @param loopname string to identity loop in statistics output
 */
template<typename IterTy, typename FunctionTy>
static inline void for_each_det(IterTy b, IterTy e, FunctionTy fn, const char* loopname = 0) {
  for_each_det(b, e, fn, fn, loopname);
}

/**
 * Deterministic execution with single operator.
 * The operator fn is used both for the prefix computation and for the
 * continuation of computation, c.f., the prefix operator version which
 * uses two different functions. The operator can distinguish between
 * the two uses by querying {@link UserContext.getLocalState()}.
 *
 * @param i initial item
 * @param fn operator
 * @param loopname string to identity loop in statistics output
 */
template<typename T, typename FunctionTy>
static inline void for_each_det(T i, FunctionTy fn, const char* loopname = 0) {
  T wl[1] = { i };
  for_each_det(&wl[0], &wl[1], fn, fn, loopname);
}

}
#endif
