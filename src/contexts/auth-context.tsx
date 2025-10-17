
'use client';

import * as React from 'react';
import type { User, Store } from '@/lib/types';
import { auth, db } from '@/lib/firebase';
import { onAuthStateChanged, signInWithEmailAndPassword, signOut } from 'firebase/auth';
import { doc, getDoc, collection, getDocs, query, where } from 'firebase/firestore';
import { useToast } from '@/hooks/use-toast';
import { useRouter } from 'next/navigation';
import { FirebaseError } from 'firebase/app';
import { FirebaseErrorListener } from '@/components/FirebaseErrorListener';

interface AuthContextType {
  currentUser: User | null;
  activeStore: Store | null;
  pradanaTokenBalance: number;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshPradanaTokenBalance: () => void;
  refreshActiveStore: () => void;
}

const AuthContext = React.createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [currentUser, setCurrentUser] = React.useState<User | null>(null);
  const [activeStore, setActiveStore] = React.useState<Store | null>(null);
  const [pradanaTokenBalance, setPradanaTokenBalance] = React.useState(0);
  const [isLoading, setIsLoading] = React.useState(true);
  const { toast } = useToast();
  const router = useRouter();
  
  const refreshActiveStore = React.useCallback(async () => {
    if (!activeStore?.id) return;
    try {
        const storeDocRef = doc(db, 'stores', activeStore.id);
        const storeDoc = await getDoc(storeDocRef);
        if (storeDoc.exists()) {
            const newStoreData = { id: storeDoc.id, ...storeDoc.data() } as Store;
            setActiveStore(newStoreData);
            setPradanaTokenBalance(newStoreData.pradanaTokenBalance || 0);
        }
    } catch (error) {
        console.error("Error refreshing active store:", error);
    }
  }, [activeStore?.id]);


  const refreshPradanaTokenBalance = React.useCallback(async () => {
    if (!activeStore) return;
    try {
      const storeDocRef = doc(db, 'stores', activeStore.id);
      const storeDoc = await getDoc(storeDocRef);
      if (storeDoc.exists()) {
        setPradanaTokenBalance(storeDoc.data()?.pradanaTokenBalance || 0);
      }
    } catch (error) {
      console.error("Error refreshing token balance:", error);
    }
  }, [activeStore]);

  const handleLogout = React.useCallback(async () => {
    await signOut(auth);
    setCurrentUser(null);
    setActiveStore(null);
    setPradanaTokenBalance(0);
  }, []);

  const handleUserSession = React.useCallback(async (user: import('firebase/auth').User | null) => {
    setIsLoading(true);
    if (user) {
      try {
        const userDocRef = doc(db, 'users', user.uid);
        const userDoc = await getDoc(userDocRef);

        if (!userDoc.exists()) {
          throw new Error(`User document not found in Firestore for UID: ${user.uid}`);
        }
        
        const userData = { id: userDoc.id, ...userDoc.data() } as User;

        if (userData.status === 'inactive') {
          throw new Error('Akun Anda tidak aktif. Silakan hubungi administrator.');
        }

        setCurrentUser(userData);

        let storeIdToLoad: string | undefined;

        if (userData.role === 'cashier' && userData.storeId) {
            storeIdToLoad = userData.storeId;
        } else if (userData.role === 'admin') {
            const storesQuery = query(collection(db, 'stores'), where('adminUids', 'array-contains', user.uid));
            const storesSnapshot = await getDocs(storesQuery);
            if (!storesSnapshot.empty) {
                storeIdToLoad = storesSnapshot.docs[0].id;
            } else {
                 throw new Error('Admin tidak terasosiasi dengan toko manapun.');
            }
        }
        
        if (!storeIdToLoad) {
           await handleLogout();
           router.push('/login');
           toast({ variant: 'destructive', title: 'Error Sesi', description: 'Tidak ada toko yang terasosiasi dengan akun Anda.' });
           setIsLoading(false); // Fix: Ensure loading is set to false
           return;
        }
        
        const storeDocRef = doc(db, 'stores', storeIdToLoad);
        const storeDoc = await getDoc(storeDocRef);

        if (!storeDoc.exists()) {
          throw new Error(`Toko dengan ID '${storeIdToLoad}' tidak ditemukan.`);
        }

        const storeData = { id: storeDoc.id, ...storeDoc.data() } as Store;

        setActiveStore(storeData);
        setPradanaTokenBalance(storeData.pradanaTokenBalance || 0);
        
      } catch (error) {
        console.error("Error handling user session:", error);
        toast({ variant: 'destructive', title: 'Error Sesi', description: (error as Error).message });
        await handleLogout();
        router.push('/login');
      } finally {
        setIsLoading(false);
      }
    } else {
      await handleLogout();
      setIsLoading(false);
    }
  }, [toast, router, handleLogout]);

  React.useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, handleUserSession);
    return () => unsubscribe();
  }, [handleUserSession]);

  const login = async (email: string, password: string) => {
    try {
      await signInWithEmailAndPassword(auth, email, password);
      // Let onAuthStateChanged handle the rest
    } catch (error) {
        let errorMessage = "Terjadi kesalahan. Silakan coba lagi.";
        if (error instanceof FirebaseError) {
            if (error.code === 'auth/user-not-found' || error.code === 'auth/wrong-password' || error.code === 'auth/invalid-credential') {
                errorMessage = "Email atau password yang Anda masukkan salah.";
            } else if (error.code === 'auth/too-many-requests') {
                errorMessage = "Terlalu banyak percobaan login. Silakan coba lagi nanti.";
            }
        }
        toast({ variant: 'destructive', title: 'Login Gagal', description: errorMessage });
        throw error;
    }
  };

  const logout = async () => {
    await handleLogout();
    router.push('/login');
    toast({
      title: 'Logout Berhasil',
      description: 'Anda telah keluar.',
    });
  };

  const value = { currentUser, activeStore, pradanaTokenBalance, isLoading, login, logout, refreshPradanaTokenBalance, refreshActiveStore };

  return (
    <AuthContext.Provider value={value}>
        {children}
        <FirebaseErrorListener />
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = React.useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
